"""
Retail Brain — Recommendation Engine
Classifies risk level and calculates replenishment quantities.
"""

import pandas as pd


# ── Risk Thresholds ────────────────────────────────────────────────────────────
HIGH_RISK_THRESHOLD   = 0.70
MEDIUM_RISK_THRESHOLD = 0.40

# ── Risk Display Configs ───────────────────────────────────────────────────────
RISK_CONFIG = {
    "High":   {"emoji": "🔴", "action": "Replenish immediately",             "badge": "danger"},
    "Medium": {"emoji": "🟡", "action": "Monitor and restock soon",           "badge": "warning"},
    "Low":    {"emoji": "🟢", "action": "No action required",                 "badge": "success"},
}


def classify_risk(probability: float) -> str:
    """Map a probability [0,1] to a risk label."""
    if probability >= HIGH_RISK_THRESHOLD:
        return "High"
    elif probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    else:
        return "Low"


def estimate_time_to_stockout(days_of_cover: float) -> str:
    """Convert days_of_cover to a human-readable time string."""
    if days_of_cover <= 0:
        return "Already stocked out"
    hours = days_of_cover * 24
    if hours < 24:
        return f"{hours:.0f} hours"
    elif days_of_cover < 7:
        return f"{days_of_cover:.1f} days"
    else:
        return f"{days_of_cover:.0f} days"


def calculate_replenishment_qty(
    stock_on_hand: float,
    reorder_point: float,
    lead_time_days: float,
    sales_velocity_7d: float,
) -> int:
    """
    Calculate how many units to order.
    Formula: cover lead-time demand + buffer to reach 2× reorder point.
    """
    lead_time_demand  = sales_velocity_7d * lead_time_days
    target_stock      = reorder_point * 2
    needed            = max(0, target_stock + lead_time_demand - stock_on_hand)
    # Round up to nearest 5 for practical ordering
    return int(max(0, -(-needed // 5) * 5))


def generate_recommendations(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the output of predict.run_inference() and adds recommendation columns.

    Returns the enhanced DataFrame with:
      - risk_level
      - risk_emoji
      - recommended_action
      - replenishment_qty
      - time_to_stockout
    """
    df = predictions_df.copy()

    df["risk_level"] = df["stockout_probability"].apply(classify_risk)
    df["risk_emoji"] = df["risk_level"].map(lambda r: RISK_CONFIG[r]["emoji"])
    df["recommended_action"] = df["risk_level"].map(
        lambda r: RISK_CONFIG[r]["action"]
    )

    df["replenishment_qty"] = df.apply(
        lambda row: calculate_replenishment_qty(
            stock_on_hand     = row.get("stock_on_hand", 0),
            reorder_point     = row.get("reorder_point", 20),
            lead_time_days    = row.get("lead_time_days", 3),
            sales_velocity_7d = row.get("sales_velocity_7d", 1),
        )
        if row["risk_level"] in ("High", "Medium") else 0,
        axis=1,
    )

    df["time_to_stockout"] = df["days_of_cover"].apply(estimate_time_to_stockout)

    return df


def get_top_risk_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top-n highest-risk products."""
    return (
        df.sort_values("stockout_probability", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    # Quick smoke test
    sample = pd.DataFrame([
        {"product_name": "Whole Milk 2L",  "stockout_probability": 0.85,
         "stock_on_hand": 5,  "reorder_point": 30, "lead_time_days": 2, "sales_velocity_7d": 8.3, "days_of_cover": 0.6},
        {"product_name": "Orange Juice 1L","stockout_probability": 0.52,
         "stock_on_hand": 20, "reorder_point": 30, "lead_time_days": 2, "sales_velocity_7d": 5.1, "days_of_cover": 3.9},
        {"product_name": "Frozen Peas 500g","stockout_probability": 0.12,
         "stock_on_hand": 80, "reorder_point": 18, "lead_time_days": 5, "sales_velocity_7d": 2.1, "days_of_cover": 38.1},
    ])

    recs = generate_recommendations(sample)
    for _, row in recs.iterrows():
        print(f"\nProduct: {row['product_name']}")
        print(f"  Risk:            {row['risk_emoji']} {row['risk_level']} ({row['stockout_probability']*100:.0f}%)")
        print(f"  Time to Stockout:{row['time_to_stockout']}")
        print(f"  Action:          {row['recommended_action']}")
        if row["replenishment_qty"] > 0:
            print(f"  Replenish:       {row['replenishment_qty']} units")
