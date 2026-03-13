"""
Retail Brain — Automated Job Scheduler
Runs daily data syncs, triggers ML inference, and dispatches alerts in the background.
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from logger import get_logger
from services.alerts import dispatcher

# Mock predictions generator just for the scheduler logic.
# In reality this interacts with the DB and predict.py
from api.predictions import _get_model, run_inference

logger = get_logger("api.scheduler")
scheduler = BackgroundScheduler()

def job_nightly_pos_sync():
    """JOB 1: Run the POS Connector sync for all stores at midnight."""
    logger.info("[CRON] Executing Nightly POS Sync...")
    # Call the connectors.demo logic here for all stores in database
    logger.info("[CRON] POS Sync Complete.")


def job_daily_stockout_inference():
    """
    JOB 2: Run the ML inference early morning (e.g., 2 AM) 
    and dispatch critical alerts before the store opens.
    """
    logger.info("[CRON] Executing Daily ML Stockout Inference...")
    
    try:
        model, meta = _get_model()
        results_df = run_inference(model, meta)
        
        # Find critical ones
        critical_df = results_df[results_df["stockout_probability"] >= 0.8]
        
        if not critical_df.empty:
            logger.warning(f"[CRON] Found {len(critical_df)} critical items. Triggering Alerts.")
            # Convert DF to list of dicts for the dispatcher
            critical_products = critical_df.to_dict('records')
            
            # Group by store_id and dispatch (simplified for demo)
            dispatcher.dispatch_critical_stockouts(
                critical_products=critical_products, 
                store_id="SBY-LON-001",
                manager_email="store_manager_london@sainsburys.mock"
            )
        else:
            logger.info("[CRON] No critical stockouts predicted for today.")
            
    except Exception as e:
        logger.error(f"[CRON] Daily Inference failed: {e}")


def init_scheduler():
    """Initialize jobs and start the background scheduler."""
    logger.info("Initializing APScheduler background jobs...")
    
    # 1. POS Data Sync (Every day at 00:05)
    scheduler.add_job(
        job_nightly_pos_sync,
        CronTrigger(hour=0, minute=5),
        id="nightly_pos_sync",
        replace_existing=True
    )
    
    # 2. ML Inference (Every day at 02:00)
    scheduler.add_job(
        job_daily_stockout_inference,
        CronTrigger(hour=2, minute=0),
        id="daily_inference",
        replace_existing=True
    )
    
    scheduler.start()
    logger.info("Background Scheduler running.")

def stop_scheduler():
    """Shut down the background scheduler gracefully."""
    scheduler.shutdown()
    logger.info("Background Scheduler stopped.")
