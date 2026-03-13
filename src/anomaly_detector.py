"""
Retail Brain — Demand Anomaly Detection Engine (Phase 5)
IsolationForest + rolling Z-score for detecting unusual demand patterns.

Detects:
  - Demand spikes: unusual positive deviations
  - Demand crashes: unusual negative deviations
  - Supply anomalies: zero sales on high-demand days

Root-cause classification:
  - weather_disruption: coincides with calendar weather events
  - promotional_surge: promo/holiday event within ±1 day
  - supply_chain_issue: inventory hit zero on the anomaly day
  - demand_shock: unexplained spike (trend break)
  - seasonal_underforecast: expected from seasonality model
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import timedelta
from typing import Optional

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from logger import get_logger

logger = get_logger(__name__)


# Events that might drive demand spikes
PROMO_EVENTS = {
    "Black Friday", "Cyber Monday", "Christmas Rush", "Christmas Eve",
    "Boxing Day", "Halloween", "Bonfire Night", "Easter Monday", "Good Friday",
}

WEATHER_DISRUPTION_EVENTS = {
    "Bank Holiday", "Spring Bank Holiday", "Summer Bank Holiday", "May Day",
}


class AnomalyDetector:
    """
    Per-product rolling anomaly detection using:
    1. Z-score (robust local baseline)
    2. IsolationForest (global multivariate anomaly scoring)

    Each anomaly is classified with a root cause for actionable insights.
    """

    Z_THRESHOLD = 2.5      # Standard deviations for Z-score detection
    IF_CONTAMINATION = 0.05  # 5% expected anomaly rate in IsolationForest

    def __init__(self):
        self.product_baselines: dict = {}  # pid → {mean, std, series}
        self.calendar_events: dict = {}    # date_str → event_name
        self.anomaly_cache: list = []      # Detected anomalies

    def fit(self, sales_df: pd.DataFrame, inventory_df: pd.DataFrame,
            calendar_df: pd.DataFrame, products_df: pd.DataFrame):
        """Fit anomaly detector on historical data."""
        logger.info("Fitting anomaly detector...")

        sales_df = sales_df.copy()
        sales_df["date"] = pd.to_datetime(sales_df["date"])

        # Build calendar event lookup
        cal = calendar_df.copy()
        cal["date"] = pd.to_datetime(cal["date"])
        # Support both 'uk_event' (real data) and 'event_name' (legacy)
        event_col = "uk_event" if "uk_event" in cal.columns else "event_name"
        self.calendar_events = dict(zip(
            cal["date"].dt.strftime("%Y-%m-%d"),
            cal[event_col].fillna("Normal")
        ))

        # Build inventory stockout lookup: (product_id, date) → bool
        inv = inventory_df.copy()
        inv["date"] = pd.to_datetime(inv["date"])
        # Support both 'stock_on_hand' (real data) and 'units_on_hand' (legacy)
        stock_col = "stock_on_hand" if "stock_on_hand" in inv.columns else "units_on_hand"
        stockout_pairs = set(
            zip(
                inv[inv[stock_col] == 0]["product_id"].astype(str),
                inv[inv[stock_col] == 0]["date"].dt.strftime("%Y-%m-%d")
            )
        )

        cat_map = dict(zip(products_df["product_id"], products_df["category"]))
        anomalies = []

        for pid in products_df["product_id"]:
            ps = sales_df[sales_df["product_id"] == pid].set_index("date")["units_sold"].sort_index()
            if len(ps) < 14:
                continue

            ps = ps.asfreq("D", fill_value=0)
            vals = ps.values.astype(float)

            # Rolling 28-day baseline
            baseline_mean = []
            baseline_std = []
            window = 28
            for i in range(len(vals)):
                start = max(0, i - window)
                hist = vals[start:i] if i > 0 else vals[:1]
                baseline_mean.append(np.mean(hist) if len(hist) > 0 else vals[i])
                baseline_std.append(max(np.std(hist), 1.0))

            baseline_mean = np.array(baseline_mean)
            baseline_std = np.array(baseline_std)
            z_scores = (vals - baseline_mean) / baseline_std

            self.product_baselines[pid] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "last_date": ps.index[-1].strftime("%Y-%m-%d"),
            }

            # IsolationForest on features: [value, z_score, day_of_week, month]
            dates = ps.index
            features = np.column_stack([
                vals,
                z_scores,
                dates.dayofweek,
                dates.month,
            ])
            if len(features) >= 20:
                clf = IsolationForest(
                    contamination=self.IF_CONTAMINATION,
                    random_state=42,
                    n_estimators=50,
                )
                if_labels = clf.fit_predict(features)
                if_scores = clf.score_samples(features)
            else:
                if_labels = np.ones(len(features), dtype=int)
                if_scores = np.zeros(len(features))

            # Collect anomalies for this product
            for i, (date, val) in enumerate(zip(dates, vals)):
                z = z_scores[i]
                is_zscore_anomaly = abs(z) > self.Z_THRESHOLD
                is_if_anomaly = if_labels[i] == -1
                if not (is_zscore_anomaly or is_if_anomaly):
                    continue
                if val == 0 and baseline_mean[i] < 1.0:
                    continue  # Ignore zero-demand products

                date_str = date.strftime("%Y-%m-%d")
                event = self.calendar_events.get(date_str, "Normal")
                had_stockout = (str(pid), date_str) in stockout_pairs

                # Determine anomaly type
                if z > self.Z_THRESHOLD:
                    anomaly_type = "demand_spike"
                elif z < -self.Z_THRESHOLD:
                    anomaly_type = "demand_crash"
                else:
                    anomaly_type = "unusual_pattern"

                # Root cause classification
                root_cause, description = self._classify_root_cause(
                    anomaly_type, event, had_stockout, z, val, baseline_mean[i]
                )

                # Severity: low / medium / high
                z_abs = abs(z)
                if z_abs >= 4.0:
                    severity = "high"
                elif z_abs >= 3.0:
                    severity = "medium"
                else:
                    severity = "low"

                anomalies.append({
                    "product_id": pid,
                    "category": cat_map.get(pid, "Unknown"),
                    "date": date_str,
                    "day_name": date.strftime("%A"),
                    "actual_units": int(val),
                    "expected_units": round(float(baseline_mean[i]), 1),
                    "z_score": round(float(z), 2),
                    "if_score": round(float(if_scores[i]), 4),
                    "anomaly_type": anomaly_type,
                    "root_cause": root_cause,
                    "description": description,
                    "severity": severity,
                    "calendar_event": event if event != "Normal" else None,
                    "stockout_on_day": had_stockout,
                })

        self.anomaly_cache = sorted(anomalies, key=lambda x: abs(x["z_score"]), reverse=True)
        logger.info("Anomaly detection complete. Found %d anomalies.", len(self.anomaly_cache))

    def _classify_root_cause(self, anomaly_type: str, event: str,
                              had_stockout: bool, z: float,
                              actual: float, expected: float):
        """Classify the root cause of an anomaly."""
        deviation_pct = round((actual - expected) / max(expected, 1) * 100)

        if anomaly_type == "demand_crash" and had_stockout:
            return "supply_chain_issue", (
                f"Stock ran out causing lost sales. Expected {expected:.0f} units but "
                f"sold only {actual:.0f} ({abs(deviation_pct)}% below baseline). "
                "Consider emergency replenishment order."
            )

        if anomaly_type == "demand_spike" and event in PROMO_EVENTS:
            return "promotional_surge", (
                f"{event} drove a {deviation_pct}% demand spike "
                f"({actual:.0f} vs {expected:.0f} expected). "
                "Ensure adequate stock for similar future events."
            )

        if anomaly_type == "demand_crash" and event in WEATHER_DISRUPTION_EVENTS:
            return "weather_disruption", (
                f"Bank holiday reduced foot traffic. Sales dropped {abs(deviation_pct)}% "
                f"({actual:.0f} vs {expected:.0f} expected). "
                "Normal pattern expected to resume next trading day."
            )

        if anomaly_type == "demand_crash" and event in PROMO_EVENTS:
            return "post_event_slump", (
                f"Post-{event} demand slump — sales {abs(deviation_pct)}% below baseline "
                f"({actual:.0f} vs {expected:.0f}). "
                "Typical rebound expected within 3-5 days."
            )

        if anomaly_type == "demand_spike":
            return "demand_shock", (
                f"Unexplained demand surge of {deviation_pct}% "
                f"({actual:.0f} vs expected {expected:.0f} units). "
                "Review local events, competitor actions, or social media trends."
            )

        if anomaly_type == "demand_crash":
            return "demand_shock", (
                f"Unexplained demand drop of {abs(deviation_pct)}% "
                f"({actual:.0f} vs expected {expected:.0f} units). "
                "Check for product availability, shelf placement, or pricing issues."
            )

        return "unusual_pattern", (
            f"Statistically unusual demand pattern detected "
            f"(z={z:.2f}, actual={actual:.0f}, expected={expected:.0f}). "
            "Monitor closely for next 7 days."
        )

    def get_recent_anomalies(self, days: int = 30, severity: Optional[str] = None,
                             limit: int = 100) -> list[dict]:
        """Return most recent anomalies, optionally filtered by severity."""
        if not self.anomaly_cache:
            return []

        all_dates = [a["date"] for a in self.anomaly_cache]
        if not all_dates:
            return []

        cutoff = (pd.Timestamp(max(all_dates)) - timedelta(days=days)).strftime("%Y-%m-%d")
        result = [a for a in self.anomaly_cache if a["date"] >= cutoff]

        if severity:
            result = [a for a in result if a["severity"] == severity]

        return result[:limit]

    def get_product_anomalies(self, pid: str, limit: int = 20) -> list[dict]:
        """Return recent anomalies for a specific product."""
        return [a for a in self.anomaly_cache if a["product_id"] == pid][:limit]

    def get_anomaly_summary(self) -> dict:
        """High-level anomaly summary for dashboard."""
        if not self.anomaly_cache:
            return {"total": 0, "high_severity": 0, "by_type": {}, "by_cause": {}}

        by_type = {}
        by_cause = {}
        by_severity = {"low": 0, "medium": 0, "high": 0}

        for a in self.anomaly_cache:
            by_type[a["anomaly_type"]] = by_type.get(a["anomaly_type"], 0) + 1
            by_cause[a["root_cause"]] = by_cause.get(a["root_cause"], 0) + 1
            by_severity[a["severity"]] = by_severity.get(a["severity"], 0) + 1

        top_affected = {}
        for a in self.anomaly_cache:
            top_affected[a["product_id"]] = top_affected.get(a["product_id"], 0) + 1

        top_products = sorted(top_affected.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total": len(self.anomaly_cache),
            "high_severity": by_severity["high"],
            "medium_severity": by_severity["medium"],
            "low_severity": by_severity["low"],
            "by_type": by_type,
            "by_cause": by_cause,
            "top_affected_products": [{"product_id": pid, "anomaly_count": cnt}
                                       for pid, cnt in top_products],
        }

    def get_active_root_causes(self) -> dict:
        """Count anomalies by root cause in the last 90 days."""
        recent = self.get_recent_anomalies(days=90)
        causes: dict = {}
        for a in recent:
            causes[a["root_cause"]] = causes.get(a["root_cause"], 0) + 1
        return dict(sorted(causes.items(), key=lambda x: x[1], reverse=True))


# ── Standalone Run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Phase 5: Demand Anomaly Detection")
    print("=" * 65)

    from data_ingestion import load_sales, load_inventory, load_calendar, load_products

    sales = load_sales()
    inventory = load_inventory()
    calendar = load_calendar()
    products = load_products()

    detector = AnomalyDetector()
    detector.fit(sales, inventory, calendar, products)

    summary = detector.get_anomaly_summary()
    print(f"\nTotal anomalies detected: {summary['total']}")
    print(f"  High severity:   {summary['high_severity']}")
    print(f"  Medium severity: {summary['medium_severity']}")
    print(f"  Low severity:    {summary['low_severity']}")
    print(f"\nBy type: {summary['by_type']}")
    print(f"By root cause: {summary['by_cause']}")

    recent = detector.get_recent_anomalies(days=30, severity="high")
    if recent:
        print(f"\nRecent high-severity anomalies (last 30 days):")
        for a in recent[:5]:
            print(f"  [{a['date']}] {a['product_id']} -- {a['anomaly_type']}")
            print(f"    Cause: {a['root_cause']}")
            print(f"    {a['description'][:100]}...")
    else:
        print("\nNo high-severity anomalies in last 30 days.")

    print("\n[OK] Anomaly detector ready.")
