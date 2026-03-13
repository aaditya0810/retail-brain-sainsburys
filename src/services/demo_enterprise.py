"""
Retail Brain — Demo for Phase 3 Data Export & Audit Logs
Simulates a manager exporting a risk report and an admin viewing logs.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from logger import get_logger

logger = get_logger("enterprise.demo")

def run_enterprise_demo():
    logger.info("--- Starting Demo for Data Export & Audit Logs ---")
    
    # Simulate a StoreManager hitting GET /api/enterprise/reports/risk/csv
    logger.info("[HTTP] GET /api/enterprise/reports/risk/csv?min_probability=0.8")
    logger.info("[SUCCESS] Streaming text/csv payload starting with:")
    logger.info("  Product ID,Product Name,Category,Stock On Hand,Reorder Point,Days of Cover,Sales Velocity 7D,Stockout Probability,Predicted Stockout (0/1)")
    logger.info("  P001,Sainsbury's Taste the Difference Pork Sausages,Meat & Fish,48.0,25.0,7.7,6.2,1.00,1")
    logger.info("  P015,Sainsbury's Fairtrade Bananas,Fresh Produce,228.0,50.0,7.6,30.0,1.00,1")
    
    # Audit log creation verification
    logger.info("\n[HTTP] GET /api/enterprise/audit")
    logger.info("[SUCCESS] Retrieved 200 OK JSON payload. Recent actions:")
    logger.info("  1. [csv_export] (User: store_manager_london@sainsburys.mock) - Exported Risk Report with min_prob > 0.8")
    logger.info("  2. [user_login] (User: store_manager_london@sainsburys.mock) - Successful login")
    logger.info("  3. [upload_sales] (Store: SBY-LON-001) - Uploaded 450 sales records")
    
    logger.info("--- Demo Run Complete ---")


if __name__ == "__main__":
    run_enterprise_demo()
