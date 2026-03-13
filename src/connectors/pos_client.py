"""
Retail Brain — POS (Point of Sale) Connectors
Abstract interfaces and mock implementations for fetching live data from Square and Lightspeed.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import random
from datetime import date, timedelta
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from logger import get_logger

logger = get_logger("connectors.pos")


class POSClient(ABC):
    """Abstract base class for all POS integrations."""
    
    def __init__(self, api_key: str, store_id: str):
        self.api_key = api_key
        self.store_id = store_id

    @abstractmethod
    def fetch_daily_sales(self, target_date: date) -> List[Dict]:
        """Fetch all sales transactions for a given date and aggregate by product_id."""
        pass

    @abstractmethod
    def fetch_inventory_levels(self) -> List[Dict]:
        """Fetch the current stock-on-hand for all products."""
        pass


class SquarePOSClient(POSClient):
    """Interface for Square POS API (Mock implementations for Sandbox testing)."""
    
    def fetch_daily_sales(self, target_date: date) -> List[Dict]:
        logger.info(f"[Square API - {self.store_id}] Requesting /v2/orders for {target_date}")
        
        # MOCK IMPLEMENTATION: Simulating an API response from Square
        # In production this would use `square.client.SquareClient(...)`
        return [
            {
                "product_id": "P001",
                "sale_date": target_date,
                "units_sold": random.randint(10, 50),
                "is_promotion": False,
                "store_id": self.store_id
            },
            {
                "product_id": "P005",
                "sale_date": target_date,
                "units_sold": random.randint(5, 30),
                "is_promotion": True,
                "promo_type": "Percentage Off",
                "store_id": self.store_id
            }
        ]

    def fetch_inventory_levels(self) -> List[Dict]:
        logger.info(f"[Square API - {self.store_id}] Requesting /v2/inventory/counts/batch-retrieve")
        
        # MOCK IMPLEMENTATION
        return [
            {
                "product_id": "P001",
                "record_date": date.today(),
                "stock_on_hand": random.randint(5, 100),
                "store_id": self.store_id
            },
            {
                "product_id": "P005",
                "record_date": date.today(),
                "stock_on_hand": random.randint(0, 20),
                "store_id": self.store_id
            }
        ]


class LightspeedPOSClient(POSClient):
    """Interface for Lightspeed Retail API (Mock implementations)."""
    
    def fetch_daily_sales(self, target_date: date) -> List[Dict]:
        logger.info(f"[Lightspeed API - {self.store_id}] Requesting /API/Account/Sale.json for {target_date}")
        
        # MOCK IMPLEMENTATION
        return [
            {
                "product_id": "P010",
                "sale_date": target_date,
                "units_sold": random.randint(20, 80),
                "is_promotion": False,
                "store_id": self.store_id
            }
        ]

    def fetch_inventory_levels(self) -> List[Dict]:
        logger.info(f"[Lightspeed API - {self.store_id}] Requesting /API/Account/Item.json")
        
        # MOCK IMPLEMENTATION
        return [
            {
                "product_id": "P010",
                "record_date": date.today(),
                "stock_on_hand": random.randint(10, 150),
                "store_id": self.store_id
            }
        ]
