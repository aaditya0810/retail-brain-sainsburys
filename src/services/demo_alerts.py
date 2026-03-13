"""
Retail Brain — Demo for Phase 3 Alert Dispatchers.
Simulates finding a critical stockout and firing Webhook / Email alerts.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from services.alerts import dispatcher
from logger import get_logger

logger = get_logger("alerts.demo")

def run_alert_demo():
    logger.info("--- Starting Simulated Alert Dispatch ---")
    
    # 1. Simulate the output of the ML model
    mock_critical_stockouts = [
        {
            "product_name": "Sainsbury's Taste the Difference Pork Sausages",
            "stock_on_hand": 48.0,
            "reorder_point": 25.0,
            "days_of_cover": 7.7,
            "stockout_probability": 1.0
        },
        {
            "product_name": "Sainsbury's Fairtrade Bananas",
            "stock_on_hand": 228.0,
            "reorder_point": 50.0,
            "days_of_cover": 7.6,
            "stockout_probability": 1.0
        }
    ]
    
    logger.info("CRON Job found 2 Critical products. Calling dispatcher.")
    
    # Set mock environment variables so the dispatcher attempts to run
    # (Without valid keys it will safely log what it *would* have sent)
    
    # 2. Dispatch
    dispatcher.dispatch_critical_stockouts(
        critical_products=mock_critical_stockouts,
        store_id="SBY-LON-001",
        manager_email="store_manager_london@sainsburys.mock"
    )
    
    logger.info("--- Alert Dispatch Complete ---")


if __name__ == "__main__":
    run_alert_demo()
