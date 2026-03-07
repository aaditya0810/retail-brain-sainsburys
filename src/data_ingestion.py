"""
Retail Brain — Sainsbury's Data Ingestion Module
Loads and merges Sainsbury's Q4 2024 raw CSVs.
"""

import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_products() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RAW_DIR, "products.csv"))


def load_sales() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "daily_sales.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_inventory() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "inventory.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_replenishment() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "replenishment.csv"))
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def load_calendar() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "calendar.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df


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

    base = pd.merge(
        data["sales"][["product_id", "date", "units_sold", "is_promotion", "uk_event"]],
        data["inventory"][["product_id", "date", "stock_on_hand"]],
        on=["product_id", "date"],
        how="inner",
    )

    cal_cols = ["date", "day_of_week", "week_of_year", "month", "is_weekend",
                "is_bank_holiday", "is_month_end", "event_multiplier",
                "is_nectar_week", "is_christmas_period"]
    base = pd.merge(base, data["calendar"][cal_cols], on="date", how="left")

    prod_cols = ["product_id", "product_name", "category", "tier",
                 "unit_price", "reorder_point", "lead_time_days"]
    base = pd.merge(base, data["products"][prod_cols], on="product_id", how="left")

    base.sort_values(["product_id", "date"], inplace=True)
    base.reset_index(drop=True, inplace=True)
    return base


if __name__ == "__main__":
    df = build_base_dataset()
    print(f"Base dataset: {df.shape}")
    print(df.head())
