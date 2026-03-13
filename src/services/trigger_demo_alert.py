"""
Retail Brain — Live Demo Trigger
Fires a WebSocket alert to the dashboard to show real-time scaling.
"""
import asyncio
import websockets
import json

async def trigger_live_alert():
    uri = "ws://localhost:8000/api/ws/alerts/SBY-LON-001"
    try:
        async with websockets.connect(uri) as websocket:
            alert = {
                "message": "🚨 URGENT: Sainsbury's Bananas running low! Predicted stockout in 4 hours.",
                "store_id": "SBY-LON-001",
                "level": "CRITICAL"
            }
            # The backend ws endpoint usually receives from internal logic, 
            # but for this demo trigger, we'll simulate a message being broadcast.
            # In our current logic, we need to call the internal manager.
            pass
    except Exception as e:
        print(f"Error: {e}. Make sure the FastAPI server is running on port 8000!")

# Since the current ws endpoint is for listening, we'll just provide a script
# that manually hits the dispatcher in the backend for the demo.
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from src.services.alerts import dispatcher

def trigger_internal():
    print("Triggering live WebSocket alert via internal dispatcher...")
    # This will trigger the formatted alert logic
    dispatcher.dispatch_critical_stockouts(
        critical_products=[{
            "product_name": "Sainsbury's Fairtrade Bananas",
            "stock_on_hand": 12.0,
            "reorder_point": 50.0,
            "days_of_cover": 0.2,
            "stockout_probability": 0.98
        }],
        store_id="SBY-LON-001"
    )
    print("Check your dashboard for the toast notification!")

if __name__ == "__main__":
    trigger_internal()
