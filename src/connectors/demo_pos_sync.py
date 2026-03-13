"""
Retail Brain — Demo specific to Phase 3 POS sync capabilities.
"""

import sys
import os
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.connectors.pos_client import SquarePOSClient
from src.logger import get_logger

logger = get_logger("connectors.demo")

try:
    from src.database import get_db_session
    from src.db_models import DailySale, Inventory
    DB_AVAILABLE = True
except ModuleNotFoundError:
    logger.warning("SQLAlchemy not installed. Running POS sync demo in DRY-RUN mode.")
    DB_AVAILABLE = False

def run_nightly_sync(store_id: str, api_key: str):
    """"Simulates a nightly cron job fetching sales & inventory from the POS."""
    
    # 1. Initialize API Client
    client = SquarePOSClient(api_key=api_key, store_id=store_id)
    
    # 2. Fetch Data
    logger.info(f"--- Starting Nightly Sync for {store_id} ---")
    today = date.today()
    sales_data = client.fetch_daily_sales(today)
    inventory_data = client.fetch_inventory_levels()
    
    # 3. Write to PostgreSQL Database (or Print in Dry Run)
    if DB_AVAILABLE:
        with get_db_session() as session:
            for sale_dict in sales_data:
                new_sale = DailySale(**sale_dict)
                session.add(new_sale)
                logger.info(f"Loaded POS Sale: Product {new_sale.product_id} | Sold {new_sale.units_sold}")
                
            for inv_dict in inventory_data:
                new_inv = Inventory(**inv_dict)
                session.add(new_inv)
                logger.info(f"Loaded POS Inventory: Product {new_inv.product_id} | Stock {new_inv.stock_on_hand}")
                
            session.commit()
    else:
        logger.info("[DRY RUN] Would save the following POS Sales to Database:")
        for sale in sales_data:
            logger.info(f"  -> Product {sale['product_id']} | Sold {sale['units_sold']}")
            
        logger.info("[DRY RUN] Would save the following POS Inventory to Database:")
        for inv in inventory_data:
            logger.info(f"  -> Product {inv['product_id']} | Stock {inv['stock_on_hand']}")

    logger.info(f"--- Nightly Sync Complete for {store_id} ---")


if __name__ == "__main__":
    # Simulate a cron job running for two different stores
    run_nightly_sync("SBY-LON-001", "sq0idp-fake-key-london")
    print("\n")
    run_nightly_sync("SBY-MAN-002", "sq0idp-fake-key-manchester")
