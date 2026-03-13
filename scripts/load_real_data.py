"""
ETL Script — Load UCI Online Retail II (real UK retailer data) into Retail Brain schema.

Source: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
        https://doi.org/10.24432/C5CG6D  (CC BY 4.0)

This script:
  1. Fetches ~1M real transactions from a UK online retailer (2009-2011)
  2. Filters to UK-only, positive transactions
  3. Maps real stock codes → Sainsbury's-style product IDs and categories
  4. Outputs: products.csv, daily_sales.csv, inventory.csv, calendar.csv, replenishment.csv
"""

import os
import sys
import hashlib
import random
from datetime import timedelta

import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
STORE_ID = "SBY-LON-001"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Map real product descriptions to Sainsbury's-style categories
CATEGORY_KEYWORDS = {
    "Dairy & Eggs": ["egg", "milk", "cheese", "butter", "cream", "yogurt"],
    "Fresh Bakery": ["cake", "bread", "muffin", "cookie", "biscuit", "brownie", "cupcake"],
    "Meat & Fish": ["chicken", "meat", "fish", "salmon", "tuna", "sausage", "ham", "beef", "pork"],
    "Fresh Produce": ["flower", "plant", "herb", "garden", "seed", "rose", "daisy", "tulip"],
    "Drinks": ["water", "juice", "tea", "coffee", "mug", "cup", "bottle", "flask", "drink"],
    "Snacks": ["chocolate", "sweet", "candy", "fudge", "toffee", "snack", "popcorn", "crisp"],
    "Frozen": ["ice", "frozen", "cool", "chill"],
    "Household": ["candle", "holder", "frame", "clock", "drawer", "box", "tin", "jar", "basket",
                   "hook", "hanger", "doormat", "towel", "cloth", "tissue"],
    "Health & Beauty": ["soap", "bath", "cream", "lotion", "lip", "lavender", "perfume", "mirror"],
    "Ambient Grocery": [],  # fallback
}

TIERS = ["Sainsbury's", "Taste the Difference", "Branded", "So Good"]
TIER_WEIGHTS = [0.40, 0.20, 0.30, 0.10]

# Events spanning both 2009-2010 and 2010-2011 date ranges
UK_HOLIDAYS = {
    # 2009
    "2009-12-19": "Christmas Rush", "2009-12-20": "Christmas Rush",
    "2009-12-21": "Christmas Rush", "2009-12-22": "Christmas Rush",
    "2009-12-23": "Christmas Rush", "2009-12-24": "Christmas Eve",
    "2009-12-25": "Christmas Day", "2009-12-26": "Boxing Day",
    "2009-12-27": "Post Christmas", "2009-12-28": "Post Christmas",
    # 2010
    "2010-01-01": "New Year",
    "2010-04-02": "Good Friday", "2010-04-05": "Easter Monday",
    "2010-05-03": "May Day", "2010-05-31": "Spring Bank Holiday",
    "2010-08-30": "Summer Bank Holiday",
    "2010-10-31": "Halloween",
    "2010-11-05": "Bonfire Night",
    "2010-11-26": "Black Friday", "2010-11-29": "Cyber Monday",
    "2010-12-19": "Christmas Rush", "2010-12-20": "Christmas Rush",
    "2010-12-21": "Christmas Rush", "2010-12-22": "Christmas Rush",
    "2010-12-23": "Christmas Rush", "2010-12-24": "Christmas Eve",
    "2010-12-25": "Christmas Day", "2010-12-26": "Boxing Day",
    "2010-12-27": "Post Christmas",
    # 2011
    "2011-01-01": "New Year",
    "2011-04-22": "Good Friday", "2011-04-25": "Easter Monday",
    "2011-05-02": "May Day", "2011-05-30": "Spring Bank Holiday",
    "2011-08-29": "Summer Bank Holiday",
    "2011-10-31": "Halloween",
    "2011-11-05": "Bonfire Night",
    "2011-11-25": "Black Friday", "2011-11-28": "Cyber Monday",
    "2011-12-09": "Christmas Rush",
}

SUPPLIERS = [
    "Sainsbury's DC Hams Hall",
    "Sainsbury's DC Emerald Park",
    "Sainsbury's DC Weybridge",
    "Sainsbury's DC Basingstoke",
    "Unilever UK",
    "Nestlé UK",
    "Premier Foods",
    "Associated British Foods",
]


def classify_category(desc: str) -> str:
    """Assign a Sainsbury's-style category based on product description keywords."""
    desc_lower = desc.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return cat
    return "Ambient Grocery"


def generate_product_id(stock_code: str, idx: int) -> str:
    """Generate a stable Sainsbury's-style product ID from stock code."""
    prefix_map = {
        "Dairy & Eggs": "D",
        "Fresh Bakery": "B",
        "Meat & Fish": "M",
        "Fresh Produce": "P",
        "Drinks": "K",
        "Snacks": "S",
        "Frozen": "F",
        "Household": "H",
        "Health & Beauty": "HB",
        "Ambient Grocery": "A",
    }
    h = hashlib.md5(stock_code.encode()).hexdigest()[:4].upper()
    return f"SAI-{h}{idx:03d}"


def fetch_dataset() -> pd.DataFrame:
    """Fetch the UCI Online Retail II dataset."""
    print("📥 Fetching UCI Online Retail II dataset (1M+ real UK transactions)...")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=502)
        df = dataset.data.features.copy()
        # Column names from UCI
        df.columns = [c.strip() for c in df.columns]
        print(f"   Fetched {len(df):,} raw records")
        return df
    except Exception as e:
        print(f"   ucimlrepo failed ({e}), trying direct download...")
        url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
        import io
        import zipfile
        import urllib.request
        response = urllib.request.urlopen(url)
        z = zipfile.ZipFile(io.BytesIO(response.read()))
        xlsx_name = [n for n in z.namelist() if n.endswith(".xlsx")][0]
        # Load BOTH sheets (Year 2009-2010 and Year 2010-2011)
        sheets = pd.read_excel(z.open(xlsx_name), sheet_name=None)
        dfs = list(sheets.values())
        df = pd.concat(dfs, ignore_index=True)
        print(f"   Downloaded {len(df):,} raw records ({len(dfs)} sheets)")
        return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to valid UK transactions."""
    print("🧹 Cleaning data...")
    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "")
        if "invoice" in cl and "date" not in cl:
            col_map[c] = "InvoiceNo"
        elif "stock" in cl:
            col_map[c] = "StockCode"
        elif "desc" in cl:
            col_map[c] = "Description"
        elif "quant" in cl:
            col_map[c] = "Quantity"
        elif "date" in cl or "invoice" in cl:
            col_map[c] = "InvoiceDate"
        elif "price" in cl:
            col_map[c] = "UnitPrice"
        elif "customer" in cl:
            col_map[c] = "CustomerID"
        elif "country" in cl:
            col_map[c] = "Country"
    df = df.rename(columns=col_map)

    required = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"   Available columns: {df.columns.tolist()}")
        raise ValueError(f"Missing columns: {missing}")

    initial = len(df)

    # Filter: UK only, positive quantity & price, not cancellations
    if "Country" in df.columns:
        df = df[df["Country"] == "United Kingdom"]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df.dropna(subset=["Description"])
    df["Description"] = df["Description"].astype(str).str.strip()
    df = df[df["Description"].str.len() > 3]

    # Ensure StockCode is always string for consistent mapping
    df["StockCode"] = df["StockCode"].astype(str).str.strip()

    # Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["date"] = df["InvoiceDate"].dt.date

    print(f"   {initial:,} → {len(df):,} valid UK transactions")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Unique products: {df['StockCode'].nunique():,}")
    return df


def build_products(df: pd.DataFrame) -> pd.DataFrame:
    """Build products.csv from real product data."""
    print("📦 Building products table...")

    # Get most common description per stock code + aggregate stats
    prod_stats = df.groupby("StockCode").agg(
        Description=("Description", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]),
        unit_price=("UnitPrice", "median"),
        total_qty=("Quantity", "sum"),
        n_days=("date", "nunique"),
    ).reset_index()

    # Filter: keep products with enough data (sold on at least 10 days)
    prod_stats = prod_stats[prod_stats["n_days"] >= 10]

    # Take top 500 by total volume (enough for a good test, manageable size)
    prod_stats = prod_stats.nlargest(500, "total_qty").reset_index(drop=True)

    # Build product table
    products = []
    for idx, row in prod_stats.iterrows():
        cat = classify_category(row["Description"])
        tier = np.random.choice(TIERS, p=TIER_WEIGHTS)
        daily_demand = row["total_qty"] / max(row["n_days"], 1)
        reorder_pt = max(int(daily_demand * 3), 10)
        lead_days = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        pid = generate_product_id(str(row["StockCode"]), idx)

        products.append({
            "product_id": pid,
            "product_name": row["Description"].title()[:80],
            "category": cat,
            "tier": tier,
            "unit_price": round(row["unit_price"], 2),
            "base_demand": int(daily_demand),
            "reorder_point": reorder_pt,
            "lead_time_days": int(lead_days),
            "_stock_code": str(row["StockCode"]),  # keep for mapping
        })

    products_df = pd.DataFrame(products)
    print(f"   {len(products_df)} products across {products_df['category'].nunique()} categories")
    print(f"   Category distribution:\n{products_df['category'].value_counts().to_string()}")
    return products_df


def build_daily_sales(df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    """Build daily_sales.csv from real transaction data."""
    print("🛒 Building daily sales table...")

    stock_to_pid = dict(zip(products_df["_stock_code"], products_df["product_id"]))

    # Aggregate daily by stock code
    daily = (
        df[df["StockCode"].isin(stock_to_pid.keys())]
        .groupby(["StockCode", "date"])
        .agg(units_sold=("Quantity", "sum"))
        .reset_index()
    )
    daily["product_id"] = daily["StockCode"].map(stock_to_pid)
    daily["store_id"] = STORE_ID
    daily["date"] = pd.to_datetime(daily["date"])

    # Simulate promotions: ~15% of days per product
    daily["is_promotion"] = np.random.binomial(1, 0.15, size=len(daily))
    daily.loc[daily["is_promotion"] == 1, "promo_type"] = "Nectar Price"
    daily.loc[daily["is_promotion"] == 0, "promo_type"] = np.nan

    # Promo uplift: +20-40% on promo days (realistic for grocery)
    promo_mask = daily["is_promotion"] == 1
    uplift = np.random.uniform(1.20, 1.40, size=promo_mask.sum())
    daily.loc[promo_mask, "units_sold"] = (daily.loc[promo_mask, "units_sold"] * uplift).round(2)

    # Map UK events
    daily["date_str"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily["uk_event"] = daily["date_str"].map(UK_HOLIDAYS).fillna("Normal")
    daily = daily.drop(columns=["date_str"])

    # Sort and add sale_id
    daily = daily.sort_values(["product_id", "date"]).reset_index(drop=True)
    daily["sale_id"] = range(1, len(daily) + 1)
    daily["units_sold"] = daily["units_sold"].round(2)
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

    cols = ["sale_id", "product_id", "store_id", "date", "units_sold", "is_promotion", "promo_type", "uk_event"]
    daily = daily[cols]

    print(f"   {len(daily):,} daily sales records")
    print(f"   Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"   Promo days: {(daily['is_promotion']==1).sum():,} ({(daily['is_promotion']==1).mean()*100:.1f}%)")
    return daily


def build_inventory(daily_sales: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    """Build inventory.csv by simulating realistic stock levels with actual stockouts."""
    print("📊 Building inventory table...")

    pid_reorder = dict(zip(products_df["product_id"], products_df["reorder_point"]))
    pid_lead = dict(zip(products_df["product_id"], products_df["lead_time_days"]))
    all_dates = sorted(daily_sales["date"].unique())
    records = []

    for pid in products_df["product_id"]:
        reorder = pid_reorder[pid]
        lead = pid_lead[pid]
        stock = reorder * np.random.uniform(1.0, 3.0)  # modest starting stock
        pid_sales = daily_sales[daily_sales["product_id"] == pid].set_index("date")["units_sold"]
        days_since_order = 999
        pending_delivery = 0
        delivery_eta = 0

        for d in all_dates:
            sold = pid_sales.get(d, 0)
            stock = max(stock - sold, 0)
            days_since_order += 1

            # Receive pending delivery
            if pending_delivery > 0 and delivery_eta <= 0:
                # 10% chance of partial delivery, 5% chance of no-show
                roll = np.random.random()
                if roll < 0.05:
                    pass  # delivery lost / delayed further
                elif roll < 0.15:
                    stock += pending_delivery * np.random.uniform(0.5, 0.8)
                else:
                    stock += pending_delivery
                pending_delivery = 0
            delivery_eta -= 1

            # Place order if below reorder point — but with realistic delays
            if stock < reorder and days_since_order > 3 and pending_delivery == 0:
                # 20% chance supplier is OOS or order gets delayed
                if np.random.random() > 0.20:
                    pending_delivery = reorder * np.random.uniform(1.5, 3.0)
                    delivery_eta = lead + np.random.randint(0, 3)  # lead time + variability
                days_since_order = 0

            records.append({
                "product_id": pid,
                "store_id": STORE_ID,
                "date": d,
                "stock_on_hand": round(stock, 2),
                "reorder_point": reorder,
            })

    inv_df = pd.DataFrame(records)
    print(f"   {len(inv_df):,} inventory records")
    zero_stock = (inv_df["stock_on_hand"] == 0).sum()
    print(f"   Stockout instances: {zero_stock:,} ({zero_stock/len(inv_df)*100:.1f}%)")
    return inv_df


def build_calendar(daily_sales: pd.DataFrame) -> pd.DataFrame:
    """Build calendar.csv with UK holiday and event metadata."""
    print("📅 Building calendar table...")

    dates = pd.to_datetime(sorted(daily_sales["date"].unique()))
    records = []

    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        event = UK_HOLIDAYS.get(ds, "Normal")
        is_holiday = 1 if event != "Normal" and "Christmas Rush" not in event else 0
        is_christmas = 1 if "Christmas" in event or "Boxing" in event or d.month == 12 and d.day >= 15 else 0

        # Event demand multiplier
        mult = 1.0
        if "Black Friday" in event or "Cyber Monday" in event:
            mult = 1.35
        elif "Christmas" in event:
            mult = 1.50
        elif "Boxing" in event:
            mult = 1.25
        elif event != "Normal":
            mult = 1.15

        records.append({
            "date": ds,
            "day_of_week": d.dayofweek,
            "day_name": d.strftime("%A"),
            "week_of_year": d.isocalendar()[1],
            "month": d.month,
            "is_weekend": 1 if d.dayofweek >= 5 else 0,
            "is_bank_holiday": is_holiday,
            "is_month_end": 1 if d == d + pd.offsets.MonthEnd(0) else 0,
            "uk_event": event,
            "event_multiplier": mult,
            "is_nectar_week": 1 if d.isocalendar()[1] % 2 == 0 else 0,
            "is_christmas_period": is_christmas,
        })

    cal_df = pd.DataFrame(records)
    print(f"   {len(cal_df)} calendar days")
    print(f"   Events: {cal_df[cal_df['uk_event'] != 'Normal']['uk_event'].nunique()} unique events")
    return cal_df


def build_replenishment(daily_sales: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    """Build replenishment.csv — simulated purchase orders based on real demand patterns."""
    print("🚚 Building replenishment table...")

    pid_lead = dict(zip(products_df["product_id"], products_df["lead_time_days"]))
    pid_reorder = dict(zip(products_df["product_id"], products_df["reorder_point"]))
    all_dates = sorted(pd.to_datetime(daily_sales["date"].unique()))
    records = []

    for pid in products_df["product_id"]:
        reorder = pid_reorder[pid]
        lead = pid_lead[pid]
        last_order = None

        for d in all_dates:
            # Order roughly every 7-14 days
            if last_order is None or (d - last_order).days >= np.random.randint(7, 15):
                order_qty = int(reorder * np.random.uniform(2.0, 4.0))
                expected = d + timedelta(days=lead)
                received = order_qty if np.random.random() > 0.05 else int(order_qty * np.random.uniform(0.7, 0.95))
                status = "received" if expected <= all_dates[-1] else "in_transit"

                records.append({
                    "product_id": pid,
                    "store_id": STORE_ID,
                    "order_date": d.strftime("%Y-%m-%d"),
                    "expected_date": expected.strftime("%Y-%m-%d"),
                    "units_ordered": order_qty,
                    "units_received": received if status == "received" else 0,
                    "supplier": random.choice(SUPPLIERS),
                    "status": status,
                })
                last_order = d

    repl_df = pd.DataFrame(records)
    print(f"   {len(repl_df):,} replenishment orders")
    shortfall = repl_df[repl_df["units_received"] < repl_df["units_ordered"]]
    print(f"   Short deliveries: {len(shortfall):,} ({len(shortfall)/len(repl_df)*100:.1f}%)")
    return repl_df


def main():
    print("=" * 70)
    print("  Retail Brain — Real Data ETL (UCI Online Retail II)")
    print("=" * 70)

    # 1. Fetch
    raw_df = fetch_dataset()

    # 2. Clean
    clean_df = clean_transactions(raw_df)

    # 3. Build products
    products_df = build_products(clean_df)

    # 4. Build daily sales
    daily_sales = build_daily_sales(clean_df, products_df)

    # 5. Build inventory
    inventory = build_inventory(daily_sales, products_df)

    # 6. Build calendar
    calendar = build_calendar(daily_sales)

    # 7. Build replenishment
    replenishment = build_replenishment(daily_sales, products_df)

    # ── Save all CSVs ──────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Drop internal mapping column
    products_out = products_df.drop(columns=["_stock_code"])

    files = {
        "products.csv": products_out,
        "daily_sales.csv": daily_sales,
        "inventory.csv": inventory,
        "calendar.csv": calendar,
        "replenishment.csv": replenishment,
    }

    print("\n💾 Saving to", OUTPUT_DIR)
    for fname, data in files.items():
        path = os.path.join(OUTPUT_DIR, fname)
        data.to_csv(path, index=False)
        print(f"   ✅ {fname}: {len(data):,} rows, {os.path.getsize(path) / 1024:.0f} KB")

    print("\n" + "=" * 70)
    print("  ✅ DONE — Real retail data loaded successfully!")
    print(f"  Products: {len(products_out):,}")
    print(f"  Daily Sales: {len(daily_sales):,}")
    print(f"  Inventory: {len(inventory):,}")
    print(f"  Calendar: {len(calendar)}")
    print(f"  Replenishment: {len(replenishment):,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
