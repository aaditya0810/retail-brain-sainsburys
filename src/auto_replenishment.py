"""
Retail Brain × Sainsbury's — Phase 4: Automated Replenishment Engine
Generates proposed purchase orders with cost-benefit analysis.

Instead of just saying "Risk: 90%", the system now outputs:
  "Order 15 units of Bananas now to avoid a £42.50 gap."

The optimization loop:
1. Predict demand (using ML model + external factors + elasticity)
2. Calculate optimal order quantity
3. Run cost-benefit analysis (holding cost vs lost revenue)
4. Generate a prioritised purchase order

Usage:
    from auto_replenishment import ReplenishmentEngine, generate_purchase_orders
"""

import os
import sys
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from logger import get_logger

logger = get_logger("auto_replenishment")


# ── Sainsbury's Supply Chain Cost Parameters ──────────────────────────────────
# These are realistic estimates for UK grocery retail

COST_PARAMS = {
    # Holding cost as % of product value per day
    "holding_cost_pct_per_day": {
        "Dairy & Eggs":     0.015,   # 1.5%/day — perishable, high waste
        "Fresh Bakery":     0.020,   # 2.0%/day — very short shelf life
        "Meat & Fish":      0.018,
        "Fresh Produce":    0.015,
        "Drinks":           0.003,   # 0.3%/day — long shelf life
        "Ambient Grocery":  0.001,   # 0.1%/day — very stable
        "Snacks":           0.002,
        "Frozen":           0.005,   # 0.5%/day — energy cost for freezing
        "Household":        0.001,
        "Health & Beauty":  0.001,
    },
    # Average lost sale opportunity cost multiplier (accounts for customer churn)
    "lost_sale_multiplier": 1.5,   # £1 lost sale = £1.50 total impact (loyalty loss)
    # Fixed ordering cost per delivery (avg transport + admin)
    "fixed_order_cost": 15.0,      # £15 per purchase order line
    # Minimum order quantities by category
    "min_order_qty": {
        "Dairy & Eggs":     6,
        "Fresh Bakery":     12,
        "Meat & Fish":      6,
        "Fresh Produce":    10,
        "Drinks":           12,
        "Ambient Grocery":  12,
        "Snacks":           12,
        "Frozen":           6,
        "Household":        6,
        "Health & Beauty":  6,
    },
}


class ReplenishmentEngine:
    """
    Generates intelligent replenishment orders using demand forecasting
    combined with cost-benefit optimization.
    """

    def __init__(self, planning_horizon_days: int = 7,
                 service_level_target: float = 0.95):
        """
        Args:
            planning_horizon_days: How many days ahead to plan stock for
            service_level_target: Target probability of not stocking out (0.95 = 95%)
        """
        self.planning_horizon = planning_horizon_days
        self.service_level = service_level_target
        # z-score for service level (e.g., 95% → 1.645)
        from scipy.stats import norm
        self.safety_factor = norm.ppf(service_level_target)

    def calculate_optimal_order(self, row: dict) -> dict:
        """
        Calculate the optimal replenishment quantity for a single product.

        Uses Economic Order Quantity (EOQ) principles adapted for retail:
        - Forecast demand over planning horizon + lead time
        - Add safety stock based on demand variability
        - Subtract current stock on hand
        - Apply minimum order quantity constraints

        Returns a dict with order details and cost-benefit analysis.
        """
        # Extract product data
        stock_on_hand = row.get("stock_on_hand", 0)
        reorder_point = row.get("reorder_point", 20)
        lead_time = row.get("lead_time_days", 3)
        velocity_7d = row.get("sales_velocity_7d", 1.0)
        velocity_std = row.get("sales_std_7d", velocity_7d * 0.2)
        unit_price = row.get("unit_price", 1.0)
        category = row.get("category", "Ambient Grocery")
        stockout_prob = row.get("stockout_probability", 0.0)
        external_factor = row.get("external_demand_factor", 1.0)
        promo_multiplier = row.get("promo_demand_multiplier", 1.0)

        # ── Step 1: Forecast demand ──────────────────────────────────────────
        # Base demand adjusted for external factors and promotions
        adjusted_velocity = velocity_7d * external_factor * promo_multiplier
        cover_period = lead_time + self.planning_horizon

        expected_demand = adjusted_velocity * cover_period
        demand_std = velocity_std * np.sqrt(cover_period)

        # Safety stock = z * σ * √(lead_time + review_period)
        safety_stock = self.safety_factor * demand_std

        # ── Step 2: Calculate order quantity ─────────────────────────────────
        target_stock = expected_demand + safety_stock
        raw_order_qty = max(0, target_stock - stock_on_hand)

        # Apply minimum order quantity
        min_qty = COST_PARAMS["min_order_qty"].get(category, 6)
        if raw_order_qty > 0 and raw_order_qty < min_qty:
            raw_order_qty = min_qty

        # Round to practical case sizes (multiples of min_qty)
        if raw_order_qty > 0:
            order_qty = int(np.ceil(raw_order_qty / min_qty) * min_qty)
        else:
            order_qty = 0

        # ── Step 3: Cost-benefit analysis ────────────────────────────────────
        holding_rate = COST_PARAMS["holding_cost_pct_per_day"].get(category, 0.005)
        lost_sale_mult = COST_PARAMS["lost_sale_multiplier"]
        fixed_cost = COST_PARAMS["fixed_order_cost"]

        # Cost of ordering (holding cost + fixed cost)
        avg_holding_days = self.planning_horizon / 2  # average time stock is held
        holding_cost = order_qty * unit_price * holding_rate * avg_holding_days
        total_order_cost = holding_cost + (fixed_cost if order_qty > 0 else 0)

        # Revenue at risk if we DON'T order
        # Expected lost units = stockout_prob * expected_demand * portion_at_risk
        days_of_cover = stock_on_hand / max(adjusted_velocity, 0.01)
        shortfall_units = max(0, expected_demand - stock_on_hand)
        probability_weighted_loss = shortfall_units * stockout_prob
        lost_revenue = probability_weighted_loss * unit_price * lost_sale_mult

        # Net benefit of ordering
        net_benefit = lost_revenue - total_order_cost

        # ── Step 4: Priority scoring ─────────────────────────────────────────
        # Higher = more urgent. Considers risk, revenue impact, and days of cover
        urgency_score = (
            stockout_prob * 40 +                          # Risk weight
            min(net_benefit / max(unit_price, 0.01), 30) + # Revenue impact (capped)
            max(0, 20 - days_of_cover * 3)                 # Inventory urgency
        )

        return {
            "order_qty": order_qty,
            "expected_demand": round(expected_demand, 1),
            "safety_stock": round(safety_stock, 1),
            "target_stock": round(target_stock, 1),
            "days_of_cover_current": round(days_of_cover, 1),
            "days_of_cover_after_order": round(
                (stock_on_hand + order_qty) / max(adjusted_velocity, 0.01), 1
            ),
            "adjusted_velocity": round(adjusted_velocity, 2),
            "holding_cost": round(holding_cost, 2),
            "fixed_order_cost": round(fixed_cost if order_qty > 0 else 0, 2),
            "total_order_cost": round(total_order_cost, 2),
            "order_value": round(order_qty * unit_price, 2),
            "lost_revenue_if_no_order": round(lost_revenue, 2),
            "net_benefit": round(net_benefit, 2),
            "urgency_score": round(urgency_score, 1),
            "recommendation": _classify_recommendation(urgency_score, stockout_prob),
        }

    def generate_purchase_orders(self, predictions_df: pd.DataFrame,
                                  products_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate purchase orders for all at-risk products.

        Args:
            predictions_df: Output of predict.run_inference() enriched with
                           external factors and elasticity features
            products_df: Product metadata (for unit prices)

        Returns:
            DataFrame with one row per product needing replenishment,
            sorted by urgency_score descending.
        """
        df = predictions_df.copy()

        # Merge unit prices if products_df provided
        if products_df is not None and "unit_price" not in df.columns:
            df = df.merge(
                products_df[["product_id", "unit_price"]],
                on="product_id", how="left"
            )

        # Fill defaults for Phase 4 columns that may not exist yet
        for col, default in [
            ("external_demand_factor", 1.0),
            ("promo_demand_multiplier", 1.0),
            ("sales_std_7d", None),
        ]:
            if col not in df.columns:
                if default is not None:
                    df[col] = default
                else:
                    # Estimate std from velocity
                    df[col] = df.get("sales_velocity_7d", 1.0) * 0.2

        orders = []
        for _, row in df.iterrows():
            order = self.calculate_optimal_order(row.to_dict())
            order["product_id"] = row.get("product_id", "")
            order["product_name"] = row.get("product_name", "")
            order["category"] = row.get("category", "")
            order["stockout_probability"] = row.get("stockout_probability", 0)
            order["stock_on_hand"] = row.get("stock_on_hand", 0)
            order["unit_price"] = row.get("unit_price", 0)
            orders.append(order)

        orders_df = pd.DataFrame(orders)

        # Filter to only products that need ordering
        orders_df = orders_df[orders_df["order_qty"] > 0].copy()

        # Sort by urgency
        orders_df = orders_df.sort_values("urgency_score", ascending=False)
        orders_df.reset_index(drop=True, inplace=True)

        logger.info(
            f"Generated {len(orders_df)} purchase orders | "
            f"Total order value: £{orders_df['order_value'].sum():,.2f} | "
            f"Total revenue protected: £{orders_df['lost_revenue_if_no_order'].sum():,.2f}"
        )

        return orders_df


def _classify_recommendation(urgency_score: float, stockout_prob: float) -> str:
    """Classify the urgency of a replenishment recommendation."""
    if urgency_score > 50 or stockout_prob > 0.80:
        return "ORDER NOW"
    elif urgency_score > 30 or stockout_prob > 0.50:
        return "Order Today"
    elif urgency_score > 15:
        return "Order This Week"
    else:
        return "Monitor"


def generate_purchase_order_summary(orders_df: pd.DataFrame) -> dict:
    """
    Generate a summary of the purchase order batch for executive reporting.
    """
    if orders_df.empty:
        return {
            "total_orders": 0,
            "total_value": 0,
            "total_units": 0,
            "revenue_protected": 0,
            "net_benefit": 0,
            "priority_breakdown": {},
        }

    priority_breakdown = orders_df["recommendation"].value_counts().to_dict()

    return {
        "total_orders": len(orders_df),
        "total_units": int(orders_df["order_qty"].sum()),
        "total_value": round(orders_df["order_value"].sum(), 2),
        "total_holding_cost": round(orders_df["holding_cost"].sum(), 2),
        "revenue_protected": round(orders_df["lost_revenue_if_no_order"].sum(), 2),
        "net_benefit": round(orders_df["net_benefit"].sum(), 2),
        "priority_breakdown": priority_breakdown,
        "top_category": (
            orders_df.groupby("category")["order_value"]
            .sum().idxmax() if not orders_df.empty else "N/A"
        ),
        "avg_urgency": round(orders_df["urgency_score"].mean(), 1),
        "generated_at": datetime.now().isoformat(),
    }


def format_purchase_order_text(order: dict) -> str:
    """Format a single purchase order as human-readable text."""
    return (
        f"📦 ORDER: {order['product_name']}\n"
        f"   Quantity: {order['order_qty']} units (£{order['order_value']:.2f})\n"
        f"   Urgency: {order['recommendation']} (score: {order['urgency_score']:.0f})\n"
        f"   Current stock: {order['stock_on_hand']:.0f} → "
        f"After order: {order['days_of_cover_after_order']:.1f} days cover\n"
        f"   Holding cost: £{order['holding_cost']:.2f} | "
        f"Revenue protected: £{order['lost_revenue_if_no_order']:.2f}\n"
        f"   Net benefit: £{order['net_benefit']:.2f}"
    )


if __name__ == "__main__":
    from data_ingestion import load_products
    from predict import run_inference, load_model

    print("=" * 60)
    print("  Phase 4: Automated Replenishment Engine")
    print("=" * 60)

    model, meta = load_model()
    predictions = run_inference(model, meta)
    products = load_products()

    engine = ReplenishmentEngine(planning_horizon_days=7, service_level_target=0.95)
    orders = engine.generate_purchase_orders(predictions, products)

    summary = generate_purchase_order_summary(orders)

    print(f"\n📋 Purchase Order Summary:")
    print(f"   Total orders: {summary['total_orders']}")
    print(f"   Total units:  {summary['total_units']:,}")
    print(f"   Total value:  £{summary['total_value']:,.2f}")
    print(f"   Revenue protected: £{summary['revenue_protected']:,.2f}")
    print(f"   Net benefit:  £{summary['net_benefit']:,.2f}")
    print(f"   Priority: {summary['priority_breakdown']}")

    print(f"\n🚨 Top 10 Most Urgent Orders:")
    print("=" * 70)
    for _, order in orders.head(10).iterrows():
        print(format_purchase_order_text(order.to_dict()))
        print()
