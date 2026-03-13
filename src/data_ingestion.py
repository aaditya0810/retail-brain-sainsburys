"""
Retail Brain × Sainsbury's Data Ingestion Module
DB-first architecture with CSV fallback.
"""

import os
import io
import pandas as pd
from typing import Optional

from database import engine, get_db_session
from db_models import Product, DailySale, Inventory, Replenishment, Calendar
from logger import get_logger

logger = get_logger("data_ingestion")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


# ── Database Loaders ────────────────────────────────────────────────────────
def _load_from_db(query: str, parse_dates: list = None) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return None
        if parse_dates:
            for col in parse_dates:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        return df
    except Exception as e:
        logger.warning(f"Failed to load from DB: {e}")
        return None

def load_products_db() -> Optional[pd.DataFrame]:
    return _load_from_db("SELECT * FROM products")

def load_sales_db() -> Optional[pd.DataFrame]:
    df = _load_from_db("SELECT * FROM daily_sales", ["sale_date"])
    if df is not None:
        df = df.rename(columns={"sale_date": "date"})
    return df

def load_inventory_db() -> Optional[pd.DataFrame]:
    df = _load_from_db("SELECT * FROM inventory", ["record_date"])
    if df is not None:
        df = df.rename(columns={"record_date": "date"})
    return df

def load_replenishment_db() -> Optional[pd.DataFrame]:
    return _load_from_db("SELECT * FROM replenishments", ["order_date"])

def load_calendar_db() -> Optional[pd.DataFrame]:
    df = _load_from_db("SELECT * FROM calendar", ["cal_date"])
    if df is not None:
        df = df.rename(columns={"cal_date": "date"})
    return df


# ── CSV Loaders (Fallback) ──────────────────────────────────────────────────
def load_products_csv() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RAW_DIR, "products.csv"))

def load_sales_csv() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "daily_sales.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_inventory_csv() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "inventory.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_replenishment_csv() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "replenishment.csv"))
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df

def load_calendar_csv() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "calendar.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Smart Loaders ───────────────────────────────────────────────────────────
def load_products() -> pd.DataFrame:
    df = load_products_db()
    if df is not None:
        logger.info(f"Loaded {len(df)} products from DB")
        return df
    logger.info("Falling back to products.csv")
    return load_products_csv()

def load_sales() -> pd.DataFrame:
    df = load_sales_db()
    if df is not None:
        logger.info(f"Loaded {len(df)} sales records from DB")
        return df
    logger.info("Falling back to daily_sales.csv")
    return load_sales_csv()

def load_inventory() -> pd.DataFrame:
    df = load_inventory_db()
    if df is not None:
        logger.info(f"Loaded {len(df)} inventory records from DB")
        return df
    logger.info("Falling back to inventory.csv")
    return load_inventory_csv()

def load_replenishment() -> pd.DataFrame:
    df = load_replenishment_db()
    if df is not None:
        logger.info(f"Loaded {len(df)} replenishment records from DB")
        return df
    try:
        logger.info("Falling back to replenishment.csv")
        return load_replenishment_csv()
    except FileNotFoundError:
         logger.warning("replenishment.csv not found, returning empty df")
         return pd.DataFrame()

def load_calendar() -> pd.DataFrame:
    df = load_calendar_db()
    if df is not None:
        logger.info(f"Loaded {len(df)} calendar records from DB")
        return df
    logger.info("Falling back to calendar.csv")
    return load_calendar_csv()


def load_all() -> dict:
    return {
        "products":      load_products(),
        "sales":         load_sales(),
        "inventory":     load_inventory(),
        "replenishment": load_replenishment(),
        "calendar":      load_calendar(),
    }


def build_base_dataset() -> pd.DataFrame:
    """Merge all tables into a single per-product-per-day analysis DataFrame."""
    data = load_all()

    # Base starts with intersecting Sales and Inventory
    base = pd.merge(
        data["sales"][["product_id", "date", "units_sold", "is_promotion", "promo_type", "uk_event"]],
        data["inventory"][["product_id", "date", "stock_on_hand"]],
        on=["product_id", "date"],
        how="inner",
    )

    # Attach Calendar features
    cal_cols = ["date", "day_of_week", "week_of_year", "month", "is_weekend",
                "is_holiday", "is_month_end", "event_multiplier",
                "is_nectar_week", "is_christmas_period"]
    
    # If falling back to CSV, fix column typo in old CSV vs DB schema
    cal_df = data["calendar"].copy()
    if "is_bank_holiday" in cal_df.columns and "is_holiday" not in cal_df.columns:
         cal_df = cal_df.rename(columns={"is_bank_holiday": "is_holiday"})

    base = pd.merge(base, cal_df[cal_cols], on="date", how="left")

    # Attach Product features
    prod_cols = ["product_id", "product_name", "category", "tier",
                 "unit_price", "reorder_point", "lead_time_days"]
    base = pd.merge(base, data["products"][prod_cols], on="product_id", how="left")

    base.sort_values(["product_id", "date"], inplace=True)
    base.reset_index(drop=True, inplace=True)
    
    # Align DB schema names back to CSV script names expected by feature_engineering
    base = base.rename(columns={"is_holiday": "is_bank_holiday"})
    
    return base


if __name__ == "__main__":
    import logging
    logging.getLogger("data_ingestion").setLevel(logging.INFO)
    df = build_base_dataset()
    print(f"\nBase dataset built: {df.shape}")
    print(df.head())
