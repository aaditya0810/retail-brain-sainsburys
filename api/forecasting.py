"""
Retail Brain — Phase 5 Forecasting API
REST endpoints for demand forecasting, anomaly detection, and LLM co-pilot.

Endpoints:
  GET  /api/v5/forecasting/product/{product_id}   — Per-product demand forecast
  GET  /api/v5/forecasting/category/{category}    — Category-level rollup forecast
  GET  /api/v5/forecasting/summary                — All-product forecast summaries
  GET  /api/v5/forecasting/anomalies              — Recent demand anomalies
  GET  /api/v5/forecasting/anomalies/summary      — Anomaly overview stats
  POST /api/v5/copilot/ask                        — LLM co-pilot question
  DELETE /api/v5/copilot/history                  — Clear conversation history
  POST /api/v5/forecasting/refresh                — Retrain forecaster (admin only)
"""

import sys
import os
import time
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from logger import get_logger
from api.auth import get_current_active_user, User

logger = get_logger("api.forecasting")
router = APIRouter(prefix="/api/v5", tags=["Phase 5 — Forecasting & Co-Pilot"])

# ── Lazy-loaded singletons ──────────────────────────────────────────────────────
_forecaster = None
_anomaly_detector = None
_copilot = None
_last_refresh: float = 0.0
_REFRESH_COOLDOWN = 300  # 5 minutes between retrains


def _get_forecaster():
    global _forecaster
    if _forecaster is None:
        try:
            from forecaster import DemandForecaster
            _forecaster = DemandForecaster.load_or_train()
        except Exception as e:
            logger.error("Failed to load/train forecaster: %s", e)
            raise HTTPException(status_code=503, detail=f"Forecasting engine unavailable: {e}")
    return _forecaster


def _get_anomaly_detector():
    global _anomaly_detector
    if _anomaly_detector is None:
        try:
            from anomaly_detector import AnomalyDetector
            from data_ingestion import load_sales, load_inventory, load_calendar, load_products
            logger.info("Fitting anomaly detector (first request)...")
            det = AnomalyDetector()
            det.fit(load_sales(), load_inventory(), load_calendar(), load_products())
            _anomaly_detector = det
        except Exception as e:
            logger.error("Failed to init anomaly detector: %s", e)
            raise HTTPException(status_code=503, detail=f"Anomaly detector unavailable: {e}")
    return _anomaly_detector


def _get_copilot():
    global _copilot
    if _copilot is None:
        try:
            from copilot import RetailCopilot
            _copilot = RetailCopilot()
            # Inject latest risk scores and forecasts into co-pilot context
            _refresh_copilot_context(_copilot)
        except Exception as e:
            logger.error("Failed to init co-pilot: %s", e)
            raise HTTPException(status_code=503, detail=f"Co-Pilot unavailable: {e}")
    return _copilot


def _refresh_copilot_context(copilot):
    """Inject live data into the co-pilot RAG context."""
    try:
        from predict import run_inference, load_model
        model, meta = load_model()
        risk_df = run_inference(model, meta)  # returns a DataFrame
        # Normalise DataFrame → list of dicts for co-pilot
        normalised_risk = []
        for _, row in risk_df.iterrows():
            normalised_risk.append({
                "product_id": str(row.get("product_id", "")),
                "product_name": str(row.get("product_name", "")),
                "category": str(row.get("category", "")),
                "stockout_risk": float(row.get("stockout_probability", 0)),
                "units_on_hand": float(row.get("stock_on_hand", 0)),
                "units_on_order": float(row.get("units_on_order", 0) if "units_on_order" in row.index else 0),
                "reorder_point": float(row.get("reorder_point", 0)),
                "days_of_cover": float(row.get("days_of_cover", 0)),
            })
        copilot.set_context_data(risk_data=normalised_risk)
        logger.info("Co-pilot risk context loaded: %d products", len(normalised_risk))
    except Exception as e:
        logger.warning("Could not refresh co-pilot risk context: %s", e)

    try:
        fc = _get_forecaster()
        summaries = fc.get_all_product_summaries(horizon=30)
        copilot.set_context_data(forecast_summary=summaries)
    except Exception as e:
        logger.warning("Could not refresh co-pilot forecast context: %s", e)

    try:
        det = _get_anomaly_detector()
        anomalies = det.get_recent_anomalies(days=30)
        copilot.set_context_data(anomaly_data=anomalies)
    except Exception as e:
        logger.warning("Could not refresh co-pilot anomaly context: %s", e)


# ── Pydantic Schemas ───────────────────────────────────────────────────────────

class ForecastSummary(BaseModel):
    product_id: str
    category: str
    total_forecast_30d: float
    avg_daily_forecast: float
    peak_day: str
    peak_demand: float
    vs_historical: float
    n_training_days: int


class DailyForecast(BaseModel):
    date: str
    day_name: str
    forecast: float
    lower_90: float
    upper_90: float
    month_seasonality: float
    event: Optional[str] = None
    event_multiplier: Optional[float] = None
    horizon_day: int


class ProductForecastResponse(BaseModel):
    product_id: str
    category: str
    horizon_days: int
    start_date: str
    end_date: str
    daily_forecasts: List[DailyForecast]
    summary: dict


class AnomalyRecord(BaseModel):
    product_id: str
    category: str
    date: str
    day_name: str
    actual_units: int
    expected_units: float
    z_score: float
    if_score: float
    anomaly_type: str
    root_cause: str
    description: str
    severity: str
    calendar_event: Optional[str] = None
    stockout_on_day: bool


class CopilotRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    product_id: Optional[str] = Field(None, pattern=r"^SAI-[A-Z0-9]+$")


class CopilotResponse(BaseModel):
    answer: str
    intent: str
    model_used: str
    product_id: Optional[str]
    sources_used: List[str]


# ── Forecasting Endpoints ──────────────────────────────────────────────────────

@router.get("/forecasting/product/{product_id}", response_model=ProductForecastResponse)
async def get_product_forecast(
    product_id: str,
    horizon: int = Query(30, ge=7, le=90, description="Forecast horizon in days"),
    current_user: User = Depends(get_current_active_user),
):
    """
    Generate a demand forecast for a single product.

    - **product_id**: Product ID in format SAI-XXXXXX
    - **horizon**: Number of days to forecast (7-90)
    """
    forecaster = _get_forecaster()
    result = forecaster.forecast_product(product_id.upper(), horizon=horizon)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


@router.get("/forecasting/category/{category}")
async def get_category_forecast(
    category: str,
    horizon: int = Query(30, ge=7, le=90),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get aggregated demand forecast for an entire product category.

    Available categories: Ambient Grocery, Frozen, Drinks, Fresh Produce,
    Fresh Bakery, Household, Dairy & Eggs, Health & Beauty, Snacks, Meat & Fish
    """
    forecaster = _get_forecaster()
    category_decoded = category.replace("-", " ").replace("%20", " ")
    result = forecaster.forecast_category(category_decoded, horizon=horizon)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


@router.get("/forecasting/summary", response_model=List[ForecastSummary])
async def get_forecast_summary(
    limit: int = Query(50, ge=1, le=500),
    category: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
):
    """
    Get 30-day demand forecast summaries for all products.

    Sorted by total forecasted demand (highest first).
    """
    forecaster = _get_forecaster()
    summaries = forecaster.get_all_product_summaries(horizon=30)

    if category:
        cat_decoded = category.replace("-", " ").replace("%20", " ")
        summaries = [s for s in summaries if s["category"].lower() == cat_decoded.lower()]

    return summaries[:limit]


@router.get("/forecasting/anomalies", response_model=List[AnomalyRecord])
async def get_anomalies(
    days: int = Query(30, ge=1, le=365, description="Look-back window in days"),
    severity: Optional[str] = Query(None, pattern="^(low|medium|high)$"),
    limit: int = Query(50, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get recent demand anomalies with root-cause classifications.

    - **days**: How many days back to look
    - **severity**: Filter by severity level (low/medium/high)
    - **limit**: Maximum records to return
    """
    detector = _get_anomaly_detector()
    anomalies = detector.get_recent_anomalies(days=days, severity=severity, limit=limit)
    return anomalies


@router.get("/forecasting/anomalies/{product_id}", response_model=List[AnomalyRecord])
async def get_product_anomalies(
    product_id: str,
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
):
    """Get anomalies for a specific product."""
    detector = _get_anomaly_detector()
    return detector.get_product_anomalies(product_id.upper(), limit=limit)


@router.get("/forecasting/anomalies-summary")
async def get_anomaly_summary(
    current_user: User = Depends(get_current_active_user),
):
    """High-level anomaly statistics for the dashboard overview panel."""
    detector = _get_anomaly_detector()
    return detector.get_anomaly_summary()


# ── Co-Pilot Endpoints ─────────────────────────────────────────────────────────

@router.post("/copilot/ask", response_model=CopilotResponse)
async def copilot_ask(
    request: CopilotRequest,
    current_user: User = Depends(get_current_active_user),
):
    """
    Ask the RetailBrain Co-Pilot a question.

    The co-pilot has access to:
    - Live stockout risk scores
    - 30-day demand forecasts
    - Recent anomalies with root-cause analysis

    Uses GPT-4o-mini if OPENAI_API_KEY is configured, otherwise falls back
    to a deterministic rule-based engine.

    Example questions:
    - "What should I order this week?"
    - "Why is SAI-AB1234 at high risk?"
    - "Show me demand anomalies from last week"
    - "What's the 30-day forecast for household products?"
    """
    copilot = _get_copilot()
    result = copilot.ask(request.question, product_id=request.product_id)
    return result


@router.delete("/copilot/history")
async def clear_copilot_history(
    current_user: User = Depends(get_current_active_user),
):
    """Clear the co-pilot conversation history."""
    copilot = _get_copilot()
    copilot.clear_history()
    return {"message": "Conversation history cleared"}


@router.get("/copilot/context")
async def get_copilot_context_info(
    current_user: User = Depends(get_current_active_user),
):
    """Return metadata about what context the co-pilot currently has loaded."""
    copilot = _get_copilot()
    ctx = copilot.context_builder
    return {
        "has_risk_data": ctx._risk_cache is not None,
        "risk_products": len(ctx._risk_cache) if ctx._risk_cache else 0,
        "has_anomaly_data": ctx._anomaly_cache is not None,
        "anomaly_count": len(ctx._anomaly_cache) if ctx._anomaly_cache else 0,
        "has_forecast_data": ctx._forecast_summary is not None,
        "forecast_products": len(ctx._forecast_summary) if ctx._forecast_summary else 0,
        "model_used": "gpt-4o-mini" if copilot._has_openai else "rule-based",
        "conversation_turns": len(copilot.conversation_history) // 2,
    }


# ── Admin: Refresh / Retrain ───────────────────────────────────────────────────

@router.post("/forecasting/refresh")
async def refresh_forecaster(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
):
    """
    Trigger a background retrain of the demand forecaster.
    Admin only. Rate-limited to once every 5 minutes.
    """
    global _forecaster, _anomaly_detector, _last_refresh

    if not getattr(current_user, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")

    now = time.time()
    if now - _last_refresh < _REFRESH_COOLDOWN:
        remaining = int(_REFRESH_COOLDOWN - (now - _last_refresh))
        raise HTTPException(
            status_code=429,
            detail=f"Refresh cooldown active. Try again in {remaining}s."
        )

    _last_refresh = now

    def _retrain():
        global _forecaster, _anomaly_detector
        try:
            logger.info("Background retrain: forecaster...")
            from forecaster import DemandForecaster
            from data_ingestion import load_sales, load_products
            sales = load_sales()
            products = load_products()
            fc = DemandForecaster()
            fc.fit(sales, products)
            fc.save()
            _forecaster = fc
            logger.info("Forecaster retrain complete")
        except Exception as e:
            logger.error("Forecaster retrain failed: %s", e)

        try:
            logger.info("Background retrain: anomaly detector...")
            from anomaly_detector import AnomalyDetector
            from data_ingestion import load_sales, load_inventory, load_calendar, load_products
            det = AnomalyDetector()
            det.fit(load_sales(), load_inventory(), load_calendar(), load_products())
            _anomaly_detector = det
            logger.info("Anomaly detector retrain complete")
        except Exception as e:
            logger.error("Anomaly detector retrain failed: %s", e)

    background_tasks.add_task(_retrain)
    return {"message": "Retrain started in background", "cooldown_seconds": _REFRESH_COOLDOWN}
