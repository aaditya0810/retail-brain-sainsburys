"""
Retail Brain × Sainsbury's — Feature Engineering Pipeline
Converts the merged base dataset into ML-ready features.

IMPORTANT: Target labels use forward-looking actual stockout events,
NOT the current day's days_of_cover (which would cause data leakage).
All features are lagged by 1 day to prevent look-ahead bias.

Run: python src/feature_engineering.py
"""

import os
import sys
import pandas as pd
import numpy as np

# Allow imports from sibling modules
sys.path.insert(0, os.path.dirname(__file__))
from data_ingestion import build_base_dataset

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-product rolling and ratio features with proper temporal alignment.

    Key design decisions:
    1. All features are computed on current-day data
    2. Target labels are computed using FUTURE stock levels (forward-looking)
    3. Features are then SHIFTED by 1 day to prevent leakage
       (day T features predict day T+1 to T+3 stockout)

    Returns a new DataFrame with feature columns appended.
    """
    features_list = []
    EPS = 1e-6  # avoid division by zero

    for product_id, group in df.groupby("product_id"):
        g = group.sort_values("date").copy()

        # ── Rolling sales velocity ───────────────────────────────────────────
        g["sales_velocity_3d"]  = g["units_sold"].rolling(3,  min_periods=1).mean()
        g["sales_velocity_7d"]  = g["units_sold"].rolling(7,  min_periods=1).mean()
        g["sales_velocity_14d"] = g["units_sold"].rolling(14, min_periods=1).mean()
        g["sales_velocity_30d"] = g["units_sold"].rolling(30, min_periods=1).mean()

        # ── Sales standard deviation (demand variability) ────────────────────
        g["sales_std_7d"] = g["units_sold"].rolling(7, min_periods=2).std().fillna(0)

        # ── Coefficient of variation (normalized variability) ────────────────
        g["sales_cv_7d"] = g["sales_std_7d"] / (g["sales_velocity_7d"] + EPS)

        # ── Stock-to-sales ratio ─────────────────────────────────────────────
        g["stock_to_sales_ratio"] = g["stock_on_hand"] / (g["sales_velocity_7d"] + EPS)

        # ── Velocity trend ────────────────────────────────────────────────────
        # Positive = demand is accelerating vs the 14-day baseline
        g["velocity_trend"] = (
            (g["sales_velocity_7d"] - g["sales_velocity_14d"]) /
            (g["sales_velocity_14d"] + EPS)
        )

        # ── Cumulative sales acceleration ─────────────────────────────────────
        g["sales_acceleration"] = g["sales_velocity_7d"] - g["sales_velocity_7d"].shift(7).fillna(0)

        # ── Stock depletion rate ───────────────────────────────────────────────
        g["stock_depletion_7d"] = g["stock_on_hand"].diff(7).fillna(0) * -1
        
        # ── Stock level relative to reorder point ─────────────────────────────
        if "reorder_point" in g.columns:
            g["stock_vs_reorder"] = g["stock_on_hand"] / (g["reorder_point"] + EPS)
        else:
            g["stock_vs_reorder"] = 1.0

        # ── Promotion pressure ────────────────────────────────────────────────
        g["promo_days_last_7"] = g["is_promotion"].rolling(7, min_periods=1).sum()

        # ── Days since last replenishment (proxy via stock_on_hand increase) ──
        stock_increase = (g["stock_on_hand"].diff() > 5).astype(int)
        cumsum_no_restock = stock_increase.groupby((stock_increase != 0).cumsum()).cumcount()
        g["days_since_restock"] = cumsum_no_restock
        
        # ── Stock on hand (log-transformed for better distribution) ───────────
        g["log_stock"] = np.log1p(g["stock_on_hand"])

        # ══════════════════════════════════════════════════════════════════════
        # TARGET LABELS — Forward-looking actual stockout events
        # ══════════════════════════════════════════════════════════════════════
        # For each day, check if stock_on_hand drops to near-zero
        # within the next 1 or 3 days. This uses FUTURE data for the target
        # only, which is appropriate for supervised learning.

        # Minimum stock in the next 1 day (shift -1)
        g["min_stock_next_1d"] = (
            g["stock_on_hand"]
            .shift(-1)
            .rolling(1, min_periods=1)
            .min()
        )
        # Minimum stock in the next 3 days (shift -1, -2, -3)
        future_stock_list = []
        for offset in range(1, 4):
            future_stock_list.append(g["stock_on_hand"].shift(-offset))
        future_stock_df = pd.concat(future_stock_list, axis=1)
        g["min_stock_next_3d"] = future_stock_df.min(axis=1)

        # Stockout threshold: stock drops below 10% of reorder point or < 2 units
        stockout_threshold = 2.0
        if "reorder_point" in g.columns:
            stockout_threshold = (g["reorder_point"] * 0.1).clip(lower=2.0)

        g["stockout_24h"] = (g["min_stock_next_1d"] <= stockout_threshold).astype(int)
        g["stockout_72h"] = (g["min_stock_next_3d"] <= stockout_threshold).astype(int)

        # Clean up helper columns
        g.drop(columns=["min_stock_next_1d", "min_stock_next_3d"], inplace=True)

        features_list.append(g)

    result = pd.concat(features_list).sort_values(["product_id", "date"])
    result.reset_index(drop=True, inplace=True)

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE LAG — shift features by 1 day to prevent look-ahead bias
    # After this: features from day T predict targets from day T+1 to T+3
    # ══════════════════════════════════════════════════════════════════════════
    for col in FEATURE_COLUMNS:
        if col in result.columns and col not in CALENDAR_FEATURES:
            # Shift within each product group
            result[col] = result.groupby("product_id")[col].shift(1)

    return result


# ── Feature column definitions ─────────────────────────────────────────────────
# NOTE: days_of_cover is deliberately EXCLUDED — it directly leaks the target

# Sainsbury's specific calendar features
CALENDAR_FEATURES = [
    "is_weekend", 
    "is_bank_holiday", 
    "is_month_end", 
    "day_of_week", 
    "week_of_year", 
    "month",
    "event_multiplier",
    "is_nectar_week",
    "is_christmas_period"
]

# Phase 4: External & elasticity features (appended when available)
EXTERNAL_FEATURES = [
    "weather_multiplier",
    "local_event_multiplier",
    "external_demand_factor",
    "price_elasticity",
    "promo_demand_multiplier",
    "elasticity_adjusted_velocity",
]

FEATURE_COLUMNS = [
    # Sales velocity features
    "sales_velocity_3d",
    "sales_velocity_7d",
    "sales_velocity_14d",
    "sales_velocity_30d",
    # Demand variability
    "sales_std_7d",
    "sales_cv_7d",
    # Stock features (no days_of_cover — it leaks target)
    "stock_to_sales_ratio",
    "stock_vs_reorder",
    "log_stock",
    # Trend features
    "velocity_trend",
    "sales_acceleration",
    "stock_depletion_7d",
    # Promotion & Replenishment
    "promo_days_last_7",
    "days_since_restock",
    # Sainsbury's Custom Features
    "event_multiplier",
    "is_nectar_week",
    "is_christmas_period",
    "is_weekend",
    "is_bank_holiday",
    "is_month_end",
    "day_of_week",
    "week_of_year",
    "month",
]

# Phase 4: Dynamic features list that includes external features when available
def get_active_feature_columns(df: pd.DataFrame) -> list:
    """Return FEATURE_COLUMNS + any Phase 4 external features present in the data."""
    active = [c for c in FEATURE_COLUMNS if c in df.columns]
    for col in EXTERNAL_FEATURES:
        if col in df.columns and col not in active:
            active.append(col)
    return active


TARGET_COLUMNS = ["stockout_24h", "stockout_72h"]


def get_feature_matrix(df: pd.DataFrame):
    """Return X (features) and y (targets) for model training."""
    clean = df.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS)
    X = clean[FEATURE_COLUMNS]
    y = clean[TARGET_COLUMNS]
    return X, y, clean


if __name__ == "__main__":
    print("Loading base dataset …")
    base = build_base_dataset()
    print(f"  Base shape: {base.shape}")

    print("Computing features …")
    features = compute_features(base)
    print(f"  Feature shape: {features.shape}")

    out_path = os.path.join(PROCESSED_DIR, "features.csv")
    features.to_csv(out_path, index=False)
    print(f"\n✅ Features saved → {out_path}")

    # Quick sanity check on targets
    clean = features.dropna(subset=TARGET_COLUMNS)
    so24 = clean["stockout_24h"].mean() * 100
    so72 = clean["stockout_72h"].mean() * 100
    print(f"   Stockout-24h prevalence: {so24:.1f}%")
    print(f"   Stockout-72h prevalence: {so72:.1f}%")
    print(f"   Feature columns: {FEATURE_COLUMNS}")
    print("   ⚠️  days_of_cover excluded from features (prevents target leakage)")
