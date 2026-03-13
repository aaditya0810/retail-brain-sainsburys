"""
Retail Brain — Phase 4: Intelligence API
Advanced ML endpoints for weather impact, price elasticity, auto-replenishment,
and RL-based decision recommendations.
"""

import sys
import os
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from logger import get_logger
from database import get_db_session
from api.auth import get_current_active_user, RequireRole, User

logger = get_logger("api.intelligence")
router = APIRouter(prefix="/api/intelligence", tags=["Phase 4 — Intelligence"])


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class WeatherImpact(BaseModel):
    date: str
    condition: str
    temp_avg: float
    rain_mm: float
    impacts: list


class ElasticityResponse(BaseModel):
    product_id: str
    category: str
    elasticity: float
    confidence: str
    nectar_20pct_uplift: float


class PurchaseOrder(BaseModel):
    product_id: str
    product_name: str
    category: str
    order_qty: int
    order_value: float
    urgency_score: float
    recommendation: str
    holding_cost: float
    lost_revenue_if_no_order: float
    net_benefit: float
    days_of_cover_current: float
    days_of_cover_after_order: float


class PurchaseOrderSummary(BaseModel):
    total_orders: int
    total_units: int
    total_value: float
    revenue_protected: float
    net_benefit: float
    priority_breakdown: dict
    generated_at: str


class RLRecommendation(BaseModel):
    product_id: str
    product_name: str
    action: str
    order_qty: int
    confidence: float
    q_values: dict


# ═══════════════════════════════════════════════════════════════════════════════
# WEATHER & EVENTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/weather/forecast", response_model=List[WeatherImpact])
async def get_weather_impact(
    days: int = Query(5, ge=1, le=14),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get weather forecast with predicted impact on Sainsbury's product categories.
    Shows which categories will see demand spikes due to temperature/rain.
    """
    try:
        from external_factors import get_weather_impact_summary
        summary = get_weather_impact_summary()
        return summary[:days]
    except Exception as e:
        logger.error(f"Weather forecast failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch weather data")


@router.get("/events/upcoming")
async def get_upcoming_events(
    days: int = Query(14, ge=1, le=30),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get upcoming local events that will affect store footfall and demand.
    """
    from datetime import date, timedelta
    from external_factors import get_local_events

    today = date.today()
    events_by_date = {}
    for i in range(days):
        d = (today + timedelta(days=i)).isoformat()
        events = get_local_events(d)
        if events:
            events_by_date[d] = events

    return {
        "store_id": "SBY-LON-001",
        "period": f"{today.isoformat()} to {(today + timedelta(days=days)).isoformat()}",
        "events": events_by_date,
        "total_event_days": len(events_by_date),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE ELASTICITY
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/elasticity/categories")
async def get_category_elasticity(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get price elasticity summary by product category.
    Shows how sensitive each category is to Nectar price promotions.
    """
    try:
        from elasticity import PriceElasticityModel, get_category_elasticity_report
        model = PriceElasticityModel.load()
        if not model.product_elasticities:
            return {"message": "Elasticity model not yet trained. Run: python src/elasticity.py"}
        report = get_category_elasticity_report(model)
        return report.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Elasticity report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/elasticity/product/{product_id}")
async def get_product_elasticity(
    product_id: str,
    discount_pct: float = Query(0.20, ge=0.01, le=0.50),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get elasticity and predicted promo uplift for a specific product.
    """
    try:
        from elasticity import PriceElasticityModel
        model = PriceElasticityModel.load()
        data = model.product_elasticities.get(product_id)
        if not data:
            raise HTTPException(status_code=404, detail="Product not found in elasticity model")

        uplift = model.predict_promo_uplift(product_id, discount_pct)
        return {
            "product_id": product_id,
            **data,
            "discount_pct": discount_pct,
            **uplift,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Product elasticity lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# AUTOMATED REPLENISHMENT
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/replenishment/orders", response_model=List[PurchaseOrder])
async def get_purchase_orders(
    limit: int = Query(50, ge=1, le=500),
    min_urgency: float = Query(0.0, ge=0.0),
    current_user: User = Depends(get_current_active_user)
):
    """
    Generate automated purchase orders with cost-benefit analysis.
    Returns proposed orders sorted by urgency.
    """
    try:
        from predict import run_inference, load_model
        from data_ingestion import load_products
        from auto_replenishment import ReplenishmentEngine

        model, meta = load_model()
        predictions = run_inference(model, meta)
        products = load_products()

        engine = ReplenishmentEngine(planning_horizon_days=7, service_level_target=0.95)
        orders = engine.generate_purchase_orders(predictions, products)

        if min_urgency > 0:
            orders = orders[orders["urgency_score"] >= min_urgency]

        return orders.head(limit).to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="ML model not found. Please train first.")
    except Exception as e:
        logger.error(f"Purchase order generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/replenishment/summary", response_model=PurchaseOrderSummary)
async def get_replenishment_summary(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get an executive summary of all proposed purchase orders.
    """
    try:
        from predict import run_inference, load_model
        from data_ingestion import load_products
        from auto_replenishment import ReplenishmentEngine, generate_purchase_order_summary

        model, meta = load_model()
        predictions = run_inference(model, meta)
        products = load_products()

        engine = ReplenishmentEngine()
        orders = engine.generate_purchase_orders(predictions, products)
        summary = generate_purchase_order_summary(orders)
        return summary
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="ML model not found.")
    except Exception as e:
        logger.error(f"Replenishment summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# REINFORCEMENT LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/rl/recommend/{product_id}", response_model=RLRecommendation)
async def get_rl_recommendation(
    product_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get the RL agent's recommended action for a specific product.
    The agent considers current stock, demand velocity, weather, events,
    and promotions to suggest the optimal ordering strategy.
    """
    try:
        from rl_agent import InventoryRLAgent
        from predict import run_inference, load_model

        agent = InventoryRLAgent.load()
        if not agent.q_table:
            raise HTTPException(
                status_code=503,
                detail="RL agent not trained. Run: python src/rl_agent.py"
            )

        model, meta = load_model()
        predictions = run_inference(model, meta)

        product_row = predictions[predictions["product_id"] == product_id]
        if product_row.empty:
            raise HTTPException(status_code=404, detail="Product not found")

        state = product_row.iloc[0].to_dict()
        rec = agent.recommend_action(state)

        return {
            "product_id": product_id,
            "product_name": state.get("product_name", "Unknown"),
            "action": rec["agent_recommendation"],
            "order_qty": rec["order_qty"],
            "confidence": rec["confidence"],
            "q_values": rec["q_values"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RL recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rl/status")
async def get_rl_status(
    current_user: User = Depends(get_current_active_user)
):
    """Get the current status and training metrics of the RL agent."""
    try:
        from rl_agent import InventoryRLAgent
        agent = InventoryRLAgent.load()
        return agent.get_training_summary()
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


@router.post("/rl/train")
async def trigger_rl_training(
    n_episodes: int = Query(200, ge=50, le=1000),
    sample_products: int = Query(30, ge=5, le=100),
    current_user: User = Depends(RequireRole(["Admin"]))
):
    """
    Trigger RL agent retraining. Admin only.
    This runs synchronously — consider moving to a background task for production.
    """
    try:
        from rl_agent import train_rl_agents
        from data_ingestion import load_sales, load_products

        sales = load_sales()
        products = load_products()
        agent = train_rl_agents(sales, products, n_episodes, sample_products)

        return {
            "status": "training_complete",
            **agent.get_training_summary(),
        }
    except Exception as e:
        logger.error(f"RL training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
