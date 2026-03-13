"""
Retail Brain — Products API
Exposes Sainsbury's product data over REST.
"""

import sys
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from database import get_db_session
from db_models import Product, Inventory
from logger import get_logger
from api.auth import get_current_active_user, User

logger = get_logger("api.products")
router = APIRouter(prefix="/api/products", tags=["Products"])


# ── Schemas ────────────────────────────────────────────────────────
class ProductResponse(BaseModel):
    product_id: str
    product_name: str
    category: str
    tier: str
    unit_price: float
    reorder_point: float
    lead_time_days: int
    store_id: str
    current_stock: Optional[float] = None
    
    class Config:
        from_attributes = True


# ── Endpoints ──────────────────────────────────────────────────────
@router.get("", response_model=List[ProductResponse])
async def list_products(
    category: Optional[str] = None,
    tier: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=1000),
    session: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """List products with optional filtering by category or tier. Secured endpoint."""
    query = session.query(Product)
    
    # Store isolation - Manager/Viewer only sees their store (unless Admin/wildcard)
    if current_user.store_id and current_user.store_id != "*":
        query = query.filter(Product.store_id == current_user.store_id)
        
    if category:
        query = query.filter(Product.category == category)
    if tier:
        query = query.filter(Product.tier == tier)
        
    products = query.offset(skip).limit(limit).all()
    
    # Attach current stock (simplified)
    result = []
    for p in products:
        # Get latest stock
        latest_inv = session.query(Inventory).filter(
            Inventory.product_id == p.product_id
        ).order_by(Inventory.record_date.desc()).first()
        
        p_dict = {
            "product_id": p.product_id,
            "product_name": p.product_name,
            "category": p.category,
            "tier": p.tier,
            "unit_price": p.unit_price,
            "reorder_point": p.reorder_point,
            "lead_time_days": p.lead_time_days,
            "store_id": p.store_id,
            "current_stock": latest_inv.stock_on_hand if latest_inv else 0.0
        }
        result.append(p_dict)
        
    return result


@router.get("/{product_id}", response_model=ProductResponse)
async def get_product(
    product_id: str,
    session: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """Get a single product by product_id."""
    query = session.query(Product).filter(Product.product_id == product_id)
    
    if current_user.store_id and current_user.store_id != "*":
        query = query.filter(Product.store_id == current_user.store_id)
        
    product = query.first()
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found or access denied")
        
    latest_inv = session.query(Inventory).filter(
        Inventory.product_id == product.product_id
    ).order_by(Inventory.record_date.desc()).first()
    
    return {
        "product_id": product.product_id,
        "product_name": product.product_name,
        "category": product.category,
        "tier": product.tier,
        "unit_price": product.unit_price,
        "reorder_point": product.reorder_point,
        "lead_time_days": product.lead_time_days,
        "store_id": product.store_id,
        "current_stock": latest_inv.stock_on_hand if latest_inv else 0.0
    }
