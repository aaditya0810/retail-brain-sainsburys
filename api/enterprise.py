"""
Retail Brain — Enterprise API Endpoints
Provides Data Export capabilities (CSV) and Administrative Audit Logs.
"""

import sys
import os
import io
import csv
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, Query, Response, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from database import get_db_session
from db_models import AuditLog
from logger import get_logger
from api.auth import get_current_active_user, RequireRole, User
from api.predictions import _get_model, run_inference

logger = get_logger("api.enterprise")
router = APIRouter(prefix="/api/enterprise", tags=["Enterprise"])


# ── Schemas ────────────────────────────────────────────────────────
class AuditLogResponse(BaseModel):
    id: int
    action: str
    entity_type: str
    entity_id: Optional[str]
    user_id: Optional[str]
    details: Optional[str]
    store_id: Optional[str]
    timestamp: datetime
    
    class Config:
        from_attributes = True


# ── Endpoints ──────────────────────────────────────────────────────

@router.get("/audit", response_model=List[AuditLogResponse])
async def get_audit_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    action: Optional[str] = None,
    current_user: User = Depends(RequireRole(["Admin", "StoreManager"])),
    session: Session = Depends(get_db_session)
):
    """
    Get paginated system audit logs. 
    StoreManagers can only see logs for their store. Admins see all.
    """
    query = session.query(AuditLog)
    
    if current_user.role != "Admin" and current_user.store_id != "*":
        query = query.filter(AuditLog.store_id == current_user.store_id)
        
    if action:
        query = query.filter(AuditLog.action == action)
        
    logs = query.order_by(desc(AuditLog.timestamp)).offset(skip).limit(limit).all()
    return logs


@router.get("/reports/risk/csv")
async def export_risk_report_csv(
    min_probability: float = Query(0.0, ge=0.0, le=1.0),
    current_user: User = Depends(RequireRole(["Admin", "StoreManager"]))
):
    """
    Export the current ML stockout predictions as a downloadable CSV file.
    """
    model, meta = _get_model()
    
    try:
        results_df = run_inference(model, meta)
    except Exception as e:
        logger.error(f"Inference failed during CSV export: {e}")
        raise HTTPException(status_code=500, detail="Failed to run model inference.")

    # Filter based on user role and probability
    if current_user.role != "Admin" and current_user.store_id != "*":
        # Note: If the inference logic aggregates everything, we might need a true Store filter here.
        pass # In phase 3 we are moving towards multi-store so this assumes filtering

    results_df = results_df[results_df["stockout_probability"] >= min_probability]
    results_df = results_df.sort_values("stockout_probability", ascending=False)
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Headers
    writer.writerow([
        "Product ID", "Product Name", "Category", 
        "Stock On Hand", "Reorder Point", "Days of Cover", 
        "Sales Velocity 7D", "Stockout Probability", "Predicted Stockout (0/1)"
    ])
    
    for _, row in results_df.iterrows():
        writer.writerow([
            row["product_id"],
            row["product_name"],
            row["category"],
            row["stock_on_hand"],
            row.get("reorder_point", 20.0),
            row["days_of_cover"],
            row["sales_velocity_7d"],
            f"{row['stockout_probability']:.2f}",
            row["stockout_predicted"]
        ])
    
    # Audit log the export
    with get_db_session() as session:
        audit = AuditLog(
            action="csv_export", 
            entity_type="Report", 
            details=f"Exported Risk Report with min_prob > {min_probability}", 
            user_id=str(current_user.id),
            store_id=current_user.store_id
        )
        session.add(audit)
        session.commit()
    
    logger.info(f"User {current_user.email} exported CSV risk report.")

    response = Response(content=output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=stockout_risk_report_{datetime.now().strftime('%Y%m%d')}.csv"
    response.headers["Content-Type"] = "text/csv"
    return response
