"""
Retail Brain — SQLAlchemy ORM Models
Defines the database schema for all retail data tables.
"""

from datetime import datetime, timezone, date

from sqlalchemy import (
    String, Integer, Float, Boolean, Date, DateTime, Text,
    Index, ForeignKey, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════════════════════════
# USER (Authentication & RBAC)
# ═══════════════════════════════════════════════════════════════════════════════
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(100), nullable=True)
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="Viewer")
    store_id: Mapped[str] = mapped_column(String(20), nullable=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    def __repr__(self) -> str:
        return f"<User {self.email} ({self.role})>"


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCT
# ═══════════════════════════════════════════════════════════════════════════════
class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    product_name: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    tier: Mapped[str] = mapped_column(String(50), nullable=False, default="Sainsbury's")
    reorder_point: Mapped[float] = mapped_column(Float, nullable=False, default=20.0)
    lead_time_days: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    unit_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    store_id: Mapped[str] = mapped_column(String(20), nullable=False, default="SBY-LON-001", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    # Relationships
    sales = relationship("DailySale", back_populates="product", cascade="all, delete-orphan")
    inventory_records = relationship("Inventory", back_populates="product", cascade="all, delete-orphan")
    replenishments = relationship("Replenishment", back_populates="product", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="product", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Product {self.product_id}: {self.product_name}>"


# ═══════════════════════════════════════════════════════════════════════════════
# DAILY SALES
# ═══════════════════════════════════════════════════════════════════════════════
class DailySale(Base):
    __tablename__ = "daily_sales"
    __table_args__ = (
        UniqueConstraint("product_id", "sale_date", name="uq_sale_product_date"),
        Index("ix_sales_product_date", "product_id", "sale_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(String(20), ForeignKey("products.product_id"), nullable=False)
    sale_date: Mapped[date] = mapped_column(Date, nullable=False)
    units_sold: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_promotion: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    promo_type: Mapped[str] = mapped_column(String(50), nullable=True)
    uk_event: Mapped[str] = mapped_column(String(50), nullable=True)
    store_id: Mapped[str] = mapped_column(String(20), nullable=False, default="SBY-LON-001", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    # Relationships
    product = relationship("Product", back_populates="sales")

    def __repr__(self) -> str:
        return f"<DailySale {self.product_id} {self.sale_date}: {self.units_sold}>"


# ═══════════════════════════════════════════════════════════════════════════════
# INVENTORY
# ═══════════════════════════════════════════════════════════════════════════════
class Inventory(Base):
    __tablename__ = "inventory"
    __table_args__ = (
        UniqueConstraint("product_id", "record_date", name="uq_inv_product_date"),
        Index("ix_inv_product_date", "product_id", "record_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(String(20), ForeignKey("products.product_id"), nullable=False)
    record_date: Mapped[date] = mapped_column(Date, nullable=False)
    stock_on_hand: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    store_id: Mapped[str] = mapped_column(String(20), nullable=False, default="STORE_001", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    # Relationships
    product = relationship("Product", back_populates="inventory_records")

    def __repr__(self) -> str:
        return f"<Inventory {self.product_id} {self.record_date}: {self.stock_on_hand}>"


# ═══════════════════════════════════════════════════════════════════════════════
# REPLENISHMENT
# ═══════════════════════════════════════════════════════════════════════════════
class Replenishment(Base):
    __tablename__ = "replenishments"
    __table_args__ = (
        Index("ix_repl_product_date", "product_id", "order_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(String(20), ForeignKey("products.product_id"), nullable=False)
    order_date: Mapped[date] = mapped_column(Date, nullable=False)
    units_ordered: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    units_received: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    store_id: Mapped[str] = mapped_column(String(20), nullable=False, default="STORE_001", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    # Relationships
    product = relationship("Product", back_populates="replenishments")

    def __repr__(self) -> str:
        return f"<Replenishment {self.product_id} {self.order_date}: {self.units_ordered}>"


# ═══════════════════════════════════════════════════════════════════════════════
# CALENDAR
# ═══════════════════════════════════════════════════════════════════════════════
class Calendar(Base):
    __tablename__ = "calendar"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cal_date: Mapped[date] = mapped_column(Date, unique=True, nullable=False, index=True)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)
    day_name: Mapped[str] = mapped_column(String(20), nullable=False)
    week_of_year: Mapped[int] = mapped_column(Integer, nullable=False)
    month: Mapped[int] = mapped_column(Integer, nullable=False)
    is_weekend: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_holiday: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_month_end: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    event_multiplier: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    is_nectar_week: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_christmas_period: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    def __repr__(self) -> str:
        return f"<Calendar {self.cal_date} ({self.day_name})>"


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION (stores model outputs)
# ═══════════════════════════════════════════════════════════════════════════════
class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_pred_product_date", "product_id", "prediction_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(String(20), ForeignKey("products.product_id"), nullable=False)
    prediction_date: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    stockout_probability: Mapped[float] = mapped_column(Float, nullable=False)
    stockout_predicted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    days_of_cover: Mapped[float] = mapped_column(Float, nullable=True)
    time_to_stockout: Mapped[str] = mapped_column(String(50), nullable=True)
    replenishment_qty: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    recommended_action: Mapped[str] = mapped_column(String(200), nullable=True)
    explanation: Mapped[str] = mapped_column(Text, nullable=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=True)
    store_id: Mapped[str] = mapped_column(String(20), nullable=False, default="STORE_001", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    # Relationships
    product = relationship("Product", back_populates="predictions")

    def __repr__(self) -> str:
        return f"<Prediction {self.product_id}: {self.risk_level} ({self.stockout_probability:.0%})>"


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT LOG
# ═══════════════════════════════════════════════════════════════════════════════
class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=True)
    entity_id: Mapped[str] = mapped_column(String(50), nullable=True)
    details: Mapped[str] = mapped_column(Text, nullable=True)
    user_id: Mapped[str] = mapped_column(String(50), nullable=True)
    store_id: Mapped[str] = mapped_column(String(20), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    def __repr__(self) -> str:
        return f"<AuditLog {self.action} {self.entity_type}:{self.entity_id}>"
