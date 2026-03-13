"""
Retail Brain × Sainsbury's — Phase 4: Price Elasticity Model
Calculates price elasticity of demand to understand how promotions and
Nectar price drops actually affect sales volume.

Key insight: Not all promos are equal. A Nectar price on milk has very different
elasticity than a Nectar price on premium chocolate.

Usage:
    from elasticity import PriceElasticityModel, compute_elasticity_features
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
from logger import get_logger

logger = get_logger("elasticity")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ELASTICITY_PATH = os.path.join(MODELS_DIR, "elasticity_model.json")


# ── Category-level elasticity priors (Sainsbury's UK retail benchmarks) ────────
# More negative = more price-sensitive. Staples are less elastic than luxuries.
CATEGORY_ELASTICITY_PRIORS = {
    "Dairy & Eggs":     -0.6,   # Staple — relatively inelastic
    "Fresh Bakery":     -0.7,
    "Meat & Fish":      -0.9,   # Moderate sensitivity
    "Fresh Produce":    -0.5,   # Very staple
    "Drinks":           -1.2,   # Highly elastic — lots of substitutes
    "Ambient Grocery":  -0.8,
    "Snacks":           -1.5,   # Very elastic — discretionary
    "Frozen":           -1.0,
    "Household":        -0.4,   # Low elasticity — necessity
    "Health & Beauty":  -0.5,
}

# Tier elasticity adjustments
TIER_ELASTICITY_ADJUSTMENT = {
    "Taste the Difference": 0.3,   # Premium shoppers are less price-sensitive
    "Sainsbury's":          0.0,   # Baseline
    "Branded":             -0.2,   # Brand-loyal but compare with own-label
    "So Good":             -0.1,   # Value shoppers are more price-sensitive
}


class PriceElasticityModel:
    """
    Computes price elasticity of demand for Sainsbury's products.

    Uses a combination of:
    1. Historical promo vs non-promo sales comparison
    2. Category-level elasticity priors
    3. Tier-level adjustments

    Elasticity = (% change in quantity) / (% change in price)
    e.g., -1.5 means a 10% price drop → 15% demand increase
    """

    def __init__(self):
        self.product_elasticities = {}
        self.category_elasticities = {}

    def fit(self, sales_df: pd.DataFrame, products_df: pd.DataFrame) -> "PriceElasticityModel":
        """
        Estimate elasticity from historical sales data.
        Compares promo periods vs non-promo periods for each product.
        """
        logger.info("Fitting price elasticity model...")

        # Merge product metadata
        merged = sales_df.merge(
            products_df[["product_id", "category", "tier", "unit_price"]],
            on="product_id", how="left"
        )

        # ── Per-product elasticity estimation ────────────────────────────────
        for pid, group in merged.groupby("product_id"):
            promo = group[group["is_promotion"] == 1]["units_sold"]
            no_promo = group[group["is_promotion"] == 0]["units_sold"]

            category = group["category"].iloc[0]
            tier = group["tier"].iloc[0]

            if len(promo) >= 5 and len(no_promo) >= 10:
                # Empirical elasticity: compare average promo vs non-promo demand
                avg_promo = promo.mean()
                avg_base = no_promo.mean()

                if avg_base > 0:
                    pct_qty_change = (avg_promo - avg_base) / avg_base

                    # Assumed average Nectar price discount: 15-25%
                    assumed_price_drop = -0.20  # 20% average Nectar discount

                    elasticity = pct_qty_change / assumed_price_drop

                    # Clamp to reasonable range
                    elasticity = np.clip(elasticity, -4.0, -0.1)
                else:
                    elasticity = CATEGORY_ELASTICITY_PRIORS.get(category, -1.0)
            else:
                # Not enough promo data — use category prior + tier adjustment
                base_e = CATEGORY_ELASTICITY_PRIORS.get(category, -1.0)
                tier_adj = TIER_ELASTICITY_ADJUSTMENT.get(tier, 0.0)
                elasticity = base_e + tier_adj

            self.product_elasticities[pid] = {
                "elasticity": round(elasticity, 3),
                "category": category,
                "tier": tier,
                "promo_observations": len(promo),
                "base_observations": len(no_promo),
                "confidence": "high" if len(promo) >= 10 else (
                    "medium" if len(promo) >= 5 else "prior"
                ),
            }

        # ── Category-level aggregation ───────────────────────────────────────
        cat_elasticities = {}
        for pid, data in self.product_elasticities.items():
            cat = data["category"]
            if cat not in cat_elasticities:
                cat_elasticities[cat] = []
            cat_elasticities[cat].append(data["elasticity"])

        self.category_elasticities = {
            cat: {
                "mean_elasticity": round(np.mean(vals), 3),
                "median_elasticity": round(np.median(vals), 3),
                "std": round(np.std(vals), 3),
                "n_products": len(vals),
            }
            for cat, vals in cat_elasticities.items()
        }

        logger.info(
            f"Elasticity model fitted for {len(self.product_elasticities)} products "
            f"across {len(self.category_elasticities)} categories"
        )
        return self

    def predict_promo_uplift(self, product_id: str,
                              discount_pct: float = 0.20) -> dict:
        """
        Predict how much a given discount will increase demand for a product.

        Args:
            product_id: SKU ID
            discount_pct: Fractional discount (0.20 = 20% off)

        Returns:
            dict with predicted_uplift_pct, expected_demand_multiplier, confidence
        """
        data = self.product_elasticities.get(product_id)
        if not data:
            return {
                "predicted_uplift_pct": 0.0,
                "expected_demand_multiplier": 1.0,
                "confidence": "none",
                "elasticity": None,
            }

        elasticity = data["elasticity"]
        # Elasticity formula: %ΔQ = elasticity × %ΔP
        # discount_pct is positive (e.g., 0.20), price change is negative
        pct_qty_change = elasticity * (-discount_pct)

        return {
            "predicted_uplift_pct": round(pct_qty_change * 100, 1),
            "expected_demand_multiplier": round(1 + pct_qty_change, 3),
            "confidence": data["confidence"],
            "elasticity": elasticity,
        }

    def get_product_elasticity(self, product_id: str) -> Optional[float]:
        """Get the estimated elasticity for a single product."""
        data = self.product_elasticities.get(product_id)
        return data["elasticity"] if data else None

    def save(self, path: str = ELASTICITY_PATH):
        """Persist the elasticity model to JSON."""
        payload = {
            "product_elasticities": self.product_elasticities,
            "category_elasticities": self.category_elasticities,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Elasticity model saved to {path}")

    @classmethod
    def load(cls, path: str = ELASTICITY_PATH) -> "PriceElasticityModel":
        """Load a previously saved elasticity model."""
        model = cls()
        if os.path.exists(path):
            with open(path, "r") as f:
                payload = json.load(f)
            model.product_elasticities = payload.get("product_elasticities", {})
            model.category_elasticities = payload.get("category_elasticities", {})
            logger.info(f"Elasticity model loaded: {len(model.product_elasticities)} products")
        else:
            logger.warning(f"No elasticity model found at {path}")
        return model


def compute_elasticity_features(df: pd.DataFrame,
                                 elasticity_model: PriceElasticityModel = None) -> pd.DataFrame:
    """
    Add elasticity-derived features to the prediction DataFrame.

    New features:
    - price_elasticity: product's estimated elasticity coefficient
    - promo_demand_multiplier: expected demand uplift if promo is running
    - elasticity_adjusted_velocity: velocity * promo multiplier (if promo active)
    """
    result = df.copy()

    if elasticity_model is None:
        try:
            elasticity_model = PriceElasticityModel.load()
        except Exception:
            logger.warning("No elasticity model available — using neutral features")
            result["price_elasticity"] = -1.0
            result["promo_demand_multiplier"] = 1.0
            result["elasticity_adjusted_velocity"] = result.get("sales_velocity_7d", 0)
            return result

    elasticities = []
    promo_mults = []

    for _, row in result.iterrows():
        pid = row.get("product_id", "")
        is_promo = row.get("is_promotion", 0)

        e = elasticity_model.get_product_elasticity(pid)
        if e is None:
            e = -1.0

        elasticities.append(e)

        if is_promo:
            uplift = elasticity_model.predict_promo_uplift(pid, discount_pct=0.20)
            promo_mults.append(uplift["expected_demand_multiplier"])
        else:
            promo_mults.append(1.0)

    result["price_elasticity"] = elasticities
    result["promo_demand_multiplier"] = promo_mults

    velocity_col = "sales_velocity_7d"
    if velocity_col in result.columns:
        result["elasticity_adjusted_velocity"] = (
            result[velocity_col] * result["promo_demand_multiplier"]
        )
    else:
        result["elasticity_adjusted_velocity"] = result["promo_demand_multiplier"]

    return result


def get_category_elasticity_report(elasticity_model: PriceElasticityModel = None) -> pd.DataFrame:
    """
    Generate a summary report of price elasticity by category.
    Useful for the dashboard and business intelligence.
    """
    if elasticity_model is None:
        elasticity_model = PriceElasticityModel.load()

    rows = []
    for cat, data in elasticity_model.category_elasticities.items():
        # Interpret the elasticity
        e = data["mean_elasticity"]
        if abs(e) > 1.5:
            sensitivity = "Very Elastic"
        elif abs(e) > 1.0:
            sensitivity = "Elastic"
        elif abs(e) > 0.5:
            sensitivity = "Moderate"
        else:
            sensitivity = "Inelastic"

        # Predict impact of a standard 20% Nectar discount
        uplift_pct = round(e * -0.20 * 100, 1)

        rows.append({
            "category": cat,
            "mean_elasticity": data["mean_elasticity"],
            "sensitivity": sensitivity,
            "n_products": data["n_products"],
            "nectar_20pct_uplift": f"+{uplift_pct}%",
        })

    return pd.DataFrame(rows).sort_values("mean_elasticity")


if __name__ == "__main__":
    from data_ingestion import load_sales, load_products

    print("=" * 60)
    print("  Phase 4: Price Elasticity Analysis")
    print("=" * 60)

    sales = load_sales()
    products = load_products()

    model = PriceElasticityModel()
    model.fit(sales, products)
    model.save()

    print("\n📊 Category Elasticity Summary:")
    report = get_category_elasticity_report(model)
    print(report.to_string(index=False))

    print("\n🔍 Sample Product Predictions (20% Nectar Price):")
    sample_pids = list(model.product_elasticities.keys())[:5]
    for pid in sample_pids:
        data = model.product_elasticities[pid]
        uplift = model.predict_promo_uplift(pid, 0.20)
        print(f"  {pid} ({data['category']}): "
              f"ε={data['elasticity']:.2f} → "
              f"20% off = +{uplift['predicted_uplift_pct']:.1f}% demand "
              f"[{uplift['confidence']}]")
