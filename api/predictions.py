"""
Retail Brain — Predictions API
Returns the latest ML model inferences securely via REST.
"""

import sys
import os
import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from database import get_db_session
from logger import get_logger
from api.auth import get_current_active_user, User
from predict import run_inference, load_model

logger = get_logger("api.predictions")
router = APIRouter(prefix="/api/predictions", tags=["Predictions"])

# Cache the model to avoid reloading on every request
_MODEL_CACHE = None
_META_CACHE = None

def _get_model():
    global _MODEL_CACHE, _META_CACHE
    if _MODEL_CACHE is None:
        try:
            _MODEL_CACHE, _META_CACHE = load_model()
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="Model file not found. Please train the model first.")
    return _MODEL_CACHE, _META_CACHE

# ── Schemas ────────────────────────────────────────────────────────
class RiskResponse(BaseModel):
    product_id: str
    product_name: str
    category: str
    stock_on_hand: float
    reorder_point: float
    days_of_cover: float
    sales_velocity_7d: float
    stockout_probability: float
    stockout_predicted: int
    risk_level: str


# ── Endpoints ──────────────────────────────────────────────────────
@router.get("/risk", response_model=List[RiskResponse])
async def get_risk_report(
    limit: int = Query(50, ge=1, le=1000),
    category: Optional[str] = None,
    min_probability: float = Query(0.0, ge=0.0, le=1.0),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get the top at-risk stockouts dynamically served from the ML pipeline.
    This replaces `result.txt` for frontend integrators.
    """
    model, meta = _get_model()
    
    try:
        # Run live inference on latest snapshot
        results_df = run_inference(model, meta)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute real-time predictions")

    # Filter rules
    if category:
        results_df = results_df[results_df["category"] == category]
        
    results_df = results_df[results_df["stockout_probability"] >= min_probability]
    
    # Sort by risk (descending)
    results_df = results_df.sort_values("stockout_probability", ascending=False)
    
    # Take top N
    top_n = results_df.head(limit)
    
    # Format response
    response = []
    for _, row in top_n.iterrows():
        prob = row["stockout_probability"]
        
        # Determine risk label
        if prob >= 0.8:
            risk_level = "Critical"
        elif prob >= 0.5:
            risk_level = "High"
        elif prob >= 0.2:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        entry = {
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "category": row["category"],
            "stock_on_hand": float(row["stock_on_hand"]),
            "reorder_point": float(row.get("reorder_point", 20.0)),
            "days_of_cover": float(row["days_of_cover"]),
            "sales_velocity_7d": float(row["sales_velocity_7d"]),
            "stockout_probability": float(prob),
            "stockout_predicted": int(row["stockout_predicted"]),
            "risk_level": risk_level
        }
        response.append(entry)
        
    return response
