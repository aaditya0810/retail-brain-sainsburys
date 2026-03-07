"""
Retail Brain — Model Training
Trains Logistic Regression, Random Forest, and XGBoost models.
Evaluates on a hold-out test set and saves the best model.
Run: python src/train_model.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed — skipping XGBoost model.")

sys.path.insert(0, os.path.dirname(__file__))
from data_ingestion import build_base_dataset
from feature_engineering import compute_features, FEATURE_COLUMNS, TARGET_COLUMNS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET = "stockout_72h"   # Primary prediction target (72-hour horizon)


def build_models() -> dict:
    """Return a dict of candidate models."""
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=5,   # handles class imbalance
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
    return models


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
    }


def main():
    print("=" * 60)
    print("  Retail Brain — Model Training")
    print("=" * 60)

    # ── Load & engineer features ──────────────────────────────────────────────
    print("\n📂 Loading data …")
    base     = build_base_dataset()
    features = compute_features(base)

    clean = features.dropna(subset=FEATURE_COLUMNS + [TARGET])
    X = clean[FEATURE_COLUMNS].astype(float)
    y = clean[TARGET].astype(int)

    print(f"   Dataset: {len(X):,} rows | Features: {len(FEATURE_COLUMNS)}")
    print(f"   Target '{TARGET}' prevalence: {y.mean()*100:.1f}%")

    # ── Temporal split (last 20% of dates = test) ─────────────────────────────
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Train and evaluate ────────────────────────────────────────────────────
    models    = build_models()
    results   = {}
    best_name = None
    best_auc  = -1

    for name, model in models.items():
        print(f"\n🔧 Training {name} …")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics

        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"   F1:        {metrics['f1']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(classification_report(y_test, model.predict(X_test), zero_division=0))

        if metrics["roc_auc"] > best_auc:
            best_auc  = metrics["roc_auc"]
            best_name = name
            best_model = model

    # ── Save best model ───────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "stockout_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"\n🏆 Best model: {best_name} (ROC-AUC = {best_auc:.4f})")
    print(f"✅ Saved → {model_path}")

    # Save metadata
    meta = {
        "best_model": best_name,
        "target": TARGET,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": results,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Metadata → {meta_path}")


if __name__ == "__main__":
    main()
