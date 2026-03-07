"""
Retail Brain — Feature Engineering Pipeline
Converts the merged base dataset into ML-ready features.
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
    Compute per-product rolling and ratio features.
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

        # ── Stock-to-sales ratio ─────────────────────────────────────────────
        g["stock_to_sales_ratio"] = g["stock_on_hand"] / (g["sales_velocity_7d"] + EPS)

        # ── Days of cover ─────────────────────────────────────────────────────
        # How many days of stock remain at current sales velocity
        g["days_of_cover"] = g["stock_on_hand"] / (g["sales_velocity_7d"] + EPS)

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

        # ── Promotion pressure ────────────────────────────────────────────────
        g["promo_days_last_7"] = g["is_promotion"].rolling(7, min_periods=1).sum()

        # ── Days since last replenishment (proxy via stock_on_hand increase) ──
        stock_increase = (g["stock_on_hand"].diff() > 5).astype(int)
        cumsum_no_restock = stock_increase.groupby((stock_increase != 0).cumsum()).cumcount()
        g["days_since_restock"] = cumsum_no_restock

        # ── ⎡Target Labels⎤ ──────────────────────────────────────────────────
        # 1 if current stock cannot cover the next 1 / 3 days at current velocity
        g["stockout_24h"] = (g["days_of_cover"] < 1).astype(int)
        g["stockout_72h"] = (g["days_of_cover"] < 3).astype(int)

        features_list.append(g)

    result = pd.concat(features_list).sort_values(["product_id", "date"])
    result.reset_index(drop=True, inplace=True)
    return result


FEATURE_COLUMNS = [
    "sales_velocity_3d",
    "sales_velocity_7d",
    "sales_velocity_14d",
    "sales_velocity_30d",
    "sales_std_7d",
    "stock_to_sales_ratio",
    "days_of_cover",
    "velocity_trend",
    "sales_acceleration",
    "stock_depletion_7d",
    "promo_days_last_7",
    "days_since_restock",
    "is_weekend",
    "is_bank_holiday",
    "is_month_end",
    "day_of_week",
    "week_of_year",
    "month",
]

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
    so24 = features["stockout_24h"].mean() * 100
    so72 = features["stockout_72h"].mean() * 100
    print(f"   Stockout-24h prevalence: {so24:.1f}%")
    print(f"   Stockout-72h prevalence: {so72:.1f}%")
