"""
Retail Brain × Sainsbury's — Model Training
Trains Logistic Regression, Random Forest, and XGBoost models.
Uses TimeSeriesSplit cross-validation for proper temporal evaluation.
Saves the best model with comprehensive metadata.

Run: python src/train_model.py
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    precision_score, recall_score
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
N_CV_FOLDS = 5            # Number of TimeSeriesSplit folds


def build_models() -> dict:
    """Return a dict of candidate models."""
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            )),
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
            scale_pos_weight=5,  # handles class imbalance
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
    return models


def evaluate_fold(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a model on a single fold, returning metric dict."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
    }


def cross_validate_model(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = N_CV_FOLDS,
) -> dict:
    """
    Run TimeSeriesSplit cross-validation.

    Returns dict with per-fold metrics, mean, std, and the best-fold model.
    """
    tscv = TimeSeriesSplit(n_splits=n_folds)
    fold_metrics = []
    best_fold_auc    = -1
    best_fold_model  = None
    best_fold_number = -1

    print(f"  Cross-validating {name} with {n_folds} folds …")

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Clone model for each fold
        from sklearn.base import clone
        fold_model = clone(model)

        # Fit
        fold_model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_fold(fold_model, X_test, y_test)
        metrics["fold"] = fold_idx + 1
        metrics["train_size"] = len(X_train)
        metrics["test_size"] = len(X_test)
        fold_metrics.append(metrics)

        print(
            f"    Fold {fold_idx + 1}: AUC={metrics['roc_auc']:.4f}  "
            f"F1={metrics['f1']:.4f}  Precision={metrics['precision']:.4f}  "
            f"Recall={metrics['recall']:.4f}  (train={len(X_train):,}, test={len(X_test):,})"
        )

        if metrics["roc_auc"] > best_fold_auc:
            best_fold_auc    = metrics["roc_auc"]
            best_fold_model  = fold_model
            best_fold_number = fold_idx + 1

    # Compute aggregate stats
    auc_scores = [m["roc_auc"] for m in fold_metrics]
    f1_scores  = [m["f1"] for m in fold_metrics]

    summary = {
        "model_name": name,
        "fold_metrics": fold_metrics,
        "mean_auc":   round(np.mean(auc_scores), 4),
        "std_auc":    round(np.std(auc_scores), 4),
        "mean_f1":    round(np.mean(f1_scores), 4),
        "std_f1":     round(np.std(f1_scores), 4),
        "best_fold":  best_fold_number,
        "best_model": best_fold_model,
    }

    print(
        f"  {name} — Mean AUC: {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f} "
        f"| Mean F1: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f} "
        f"| Best fold: {best_fold_number}"
    )

    return summary


def main():
    print("=" * 60)
    print("  Retail Brain × Sainsbury's — Model Training")
    print("=" * 60)

    # ── Load & engineer features ──────────────────────────────────────────────
    t_start = time.time()
    print("\n📂 Loading data …")
    base     = build_base_dataset()
    features = compute_features(base)

    clean = features.dropna(subset=FEATURE_COLUMNS + [TARGET])
    X = clean[FEATURE_COLUMNS].astype(float)
    y = clean[TARGET].astype(int)

    print(f"   Dataset: {len(X):,} rows | Features: {len(FEATURE_COLUMNS)}")
    print(f"   Target '{TARGET}' prevalence: {y.mean() * 100:.1f}%")
    print(f"   Data prep time: {time.time() - t_start:.1f}s")

    # ── Cross-validate all models ─────────────────────────────────────────────
    models  = build_models()
    results = {}
    best_name      = None
    best_mean_auc  = -1
    best_model_obj = None

    for name, model in models.items():
        print(f"\n🔧 Training {name} …")
        t_model = time.time()

        summary = cross_validate_model(name, model, X, y)
        results[name] = {
            "mean_auc":   summary["mean_auc"],
            "std_auc":    summary["std_auc"],
            "mean_f1":    summary["mean_f1"],
            "std_f1":     summary["std_f1"],
            "best_fold":  summary["best_fold"],
            "fold_metrics": summary["fold_metrics"],
        }

        print(f"   Training time: {time.time() - t_model:.1f}s")

        if summary["mean_auc"] > best_mean_auc:
            best_mean_auc  = summary["mean_auc"]
            best_name      = name
            best_model_obj = summary["best_model"]

    # ── Retrain best model on full dataset for deployment ─────────────────────
    print(f"\n🏆 Best model: {best_name} (Mean AUC = {best_mean_auc:.4f} ± {results[best_name]['std_auc']:.4f})")
    print(f"🔄 Retraining {best_name} on full dataset for deployment …")
    
    from sklearn.base import clone
    final_model = clone(models[best_name])
    final_model.fit(X, y)

    # Final evaluation on last 20% (holdout for reference)
    split_idx = int(len(X) * 0.80)
    X_holdout = X.iloc[split_idx:]
    y_holdout = y.iloc[split_idx:]
    holdout_metrics = evaluate_fold(final_model, X_holdout, y_holdout)
    print(f"   Holdout metrics (last 20%): AUC={holdout_metrics['roc_auc']:.4f}  F1={holdout_metrics['f1']:.4f}")
    print(classification_report(y_holdout, final_model.predict(X_holdout), zero_division=0))

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "stockout_model.joblib")
    joblib.dump(final_model, model_path)
    print(f"✅ Model saved → {model_path}")

    # Save metadata
    meta = {
        "best_model":       best_name,
        "target":           TARGET,
        "feature_columns":  FEATURE_COLUMNS,
        "cv_folds":         N_CV_FOLDS,
        "cv_results":       results,
        "holdout_metrics":  holdout_metrics,
        "total_samples":    len(X),
        "target_prevalence": round(y.mean(), 4),
        "training_time_seconds": round(time.time() - t_start, 1),
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"✅ Metadata saved → {meta_path}")

    print("\n" + "═" * 60)
    print(f"  Training complete in {time.time() - t_start:.1f} seconds")
    print("═" * 60)


if __name__ == "__main__":
    main()
