"""
Retail Brain — Inference Module
Loads the trained model and runs predictions on the latest data snapshot.
Run: python src/predict.py
"""

import os
import sys
import json
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_ingestion import build_base_dataset
from feature_engineering import compute_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "stockout_model.joblib")
META_PATH   = os.path.join(MODELS_DIR, "model_metadata.json")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Please run: python src/train_model.py"
        )
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


def get_latest_snapshot(features_df: pd.DataFrame) -> pd.DataFrame:
    """Return the most-recent row per product (latest date in the dataset)."""
    return (
        features_df
        .sort_values("date")
        .groupby("product_id")
        .last()
        .reset_index()
    )


def run_inference(model=None, meta=None) -> pd.DataFrame:
    """
    Run the full inference pipeline.
    Returns a DataFrame with one row per product including predictions.
    """
    if model is None or meta is None:
        model, meta = load_model()

    feature_cols = meta["feature_columns"]

    # Build features from latest data
    base     = build_base_dataset()
    features = compute_features(base)
    snapshot = get_latest_snapshot(features)

    # Product metadata for display
    keep_cols = [
        "product_id", "product_name", "category",
        "stock_on_hand", "reorder_point", "lead_time_days",
        "days_of_cover", "sales_velocity_7d", "velocity_trend",
        "promo_days_last_7", "date",
    ]

    X = snapshot[feature_cols].fillna(0).astype(float)

    snapshot = snapshot.copy()
    snapshot["stockout_probability"] = model.predict_proba(X)[:, 1]
    snapshot["stockout_predicted"]   = model.predict(X)

    return snapshot[
        [c for c in keep_cols if c in snapshot.columns]
        + ["stockout_probability", "stockout_predicted"]
    ]


if __name__ == "__main__":
    print("Loading model …")
    model, meta = load_model()
    print(f"Model: {meta['best_model']} | Target: {meta['target']}")

    print("\nRunning inference …")
    results = run_inference(model, meta)

    # Sort by risk descending
    results_sorted = results.sort_values("stockout_probability", ascending=False)

    print("\n📊 Top 10 Products by Stockout Risk")
    print("=" * 70)
    display = results_sorted.head(10)[[
        "product_name", "stock_on_hand", "days_of_cover",
        "sales_velocity_7d", "stockout_probability"
    ]].copy()
    display.columns = ["Product", "Stock", "Days Cover", "Velocity 7d", "Risk %"]
    display["Risk %"] = (display["Risk %"] * 100).round(1).astype(str) + "%"
    display["Days Cover"] = display["Days Cover"].round(2)
    display["Velocity 7d"] = display["Velocity 7d"].round(2)
    print(display.to_string(index=False))
