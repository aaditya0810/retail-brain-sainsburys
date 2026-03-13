"""
Retail Brain — Data Upload Endpoints
Handles CSV/Excel file uploads for products, sales, and inventory data.
"""

import io
import sys
import os
from typing import Optional

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from database import get_db_session, engine
from db_models import Product, DailySale, Inventory, Replenishment, AuditLog, Base
from logger import get_logger

logger = get_logger("api.upload")
router = APIRouter(prefix="/api/upload", tags=["Data Upload"])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_upload(file: UploadFile) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a DataFrame."""
    content = file.file.read()
    filename = file.filename or ""

    if filename.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(400, f"Failed to parse Excel file: {e}")
    elif filename.endswith(".csv") or not filename:
        try:
            return pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(400, f"Failed to parse CSV file: {e}")
    else:
        raise HTTPException(400, f"Unsupported file format: {filename}. Use .csv or .xlsx")


def _validate_columns(df: pd.DataFrame, required: list[str], name: str):
    """Ensure required columns exist in the DataFrame."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            400,
            f"Missing required columns in {name}: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _log_audit(session: Session, action: str, details: str, store_id: str = "STORE_001"):
    """Write an audit log entry."""
    entry = AuditLog(
        action=action,
        entity_type="upload",
        details=details,
        store_id=store_id,
    )
    session.add(entry)


# ── Upload Endpoints ──────────────────────────────────────────────────────────

@router.post("/products")
async def upload_products(
    file: UploadFile = File(...),
    store_id: str = Query("STORE_001", description="Store identifier"),
    session: Session = Depends(get_db_session),
):
    """
    Upload product catalogue data (CSV or Excel).

    Required columns: product_id, product_name, category
    Optional columns: reorder_point, lead_time_days, unit_price
    """
    df = _read_upload(file)
    _validate_columns(df, ["product_id", "product_name", "category"], "products")

    # Set defaults for optional columns
    if "tier" not in df.columns:
        df["tier"] = "Sainsbury's"
    if "reorder_point" not in df.columns:
        df["reorder_point"] = 20.0
    if "lead_time_days" not in df.columns:
        df["lead_time_days"] = 3
    if "unit_price" not in df.columns:
        df["unit_price"] = 0.0
    df["store_id"] = store_id

    try:
        df.to_sql("products", engine, if_exists="append", index=False)
        _log_audit(session, "upload_products",
                   f"Uploaded {len(df)} products for store {store_id}",
                   store_id)
        session.commit()
        logger.info("Uploaded %d products for store %s", len(df), store_id)
    except Exception as e:
        session.rollback()
        raise HTTPException(500, f"Failed to save products: {e}")

    return {
        "status": "success",
        "message": f"Uploaded {len(df)} products",
        "store_id": store_id,
        "rows": len(df),
        "columns": list(df.columns),
    }


@router.post("/sales")
async def upload_sales(
    file: UploadFile = File(...),
    store_id: str = Query("STORE_001", description="Store identifier"),
    session: Session = Depends(get_db_session),
):
    """
    Upload daily sales data (CSV or Excel).

    Required columns: product_id, date, units_sold
    Optional columns: is_promotion
    """
    df = _read_upload(file)
    _validate_columns(df, ["product_id", "date", "units_sold"], "sales")

    # Normalize column names
    if "is_promotion" not in df.columns:
        df["is_promotion"] = 0
    if "promo_type" not in df.columns:
        df["promo_type"] = None
    if "uk_event" not in df.columns:
        df["uk_event"] = None

    # Rename for DB schema
    df = df.rename(columns={"date": "sale_date"})
    df["sale_date"] = pd.to_datetime(df["sale_date"]).dt.date
    df["store_id"] = store_id

    try:
        df.to_sql("daily_sales", engine, if_exists="append", index=False)
        _log_audit(session, "upload_sales",
                   f"Uploaded {len(df)} sales records for store {store_id}",
                   store_id)
        session.commit()
        logger.info("Uploaded %d sales records for store %s", len(df), store_id)
    except Exception as e:
        session.rollback()
        raise HTTPException(500, f"Failed to save sales data: {e}")

    return {
        "status": "success",
        "message": f"Uploaded {len(df)} sales records",
        "store_id": store_id,
        "rows": len(df),
        "date_range": {
            "from": str(df["sale_date"].min()),
            "to": str(df["sale_date"].max()),
        },
    }


@router.post("/inventory")
async def upload_inventory(
    file: UploadFile = File(...),
    store_id: str = Query("STORE_001", description="Store identifier"),
    session: Session = Depends(get_db_session),
):
    """
    Upload inventory/stock data (CSV or Excel).

    Required columns: product_id, date, stock_on_hand
    """
    df = _read_upload(file)
    _validate_columns(df, ["product_id", "date", "stock_on_hand"], "inventory")

    # Rename for DB schema
    df = df.rename(columns={"date": "record_date"})
    df["record_date"] = pd.to_datetime(df["record_date"]).dt.date
    df["store_id"] = store_id

    try:
        df.to_sql("inventory", engine, if_exists="append", index=False)
        _log_audit(session, "upload_inventory",
                   f"Uploaded {len(df)} inventory records for store {store_id}",
                   store_id)
        session.commit()
        logger.info("Uploaded %d inventory records for store %s", len(df), store_id)
    except Exception as e:
        session.rollback()
        raise HTTPException(500, f"Failed to save inventory data: {e}")

    return {
        "status": "success",
        "message": f"Uploaded {len(df)} inventory records",
        "store_id": store_id,
        "rows": len(df),
        "date_range": {
            "from": str(df["record_date"].min()),
            "to": str(df["record_date"].max()),
        },
    }


@router.get("/status")
async def upload_status(session: Session = Depends(get_db_session)):
    """Get current data upload status — how many records exist in each table."""
    from sqlalchemy import func, select

    counts = {}
    for table_name, model in [
        ("products", Product),
        ("daily_sales", DailySale),
        ("inventory", Inventory),
    ]:
        try:
            result = session.execute(select(func.count()).select_from(model)).scalar()
            counts[table_name] = result or 0
        except Exception:
            counts[table_name] = 0

    return {
        "status": "success",
        "record_counts": counts,
        "data_available": all(v > 0 for v in counts.values()),
    }
