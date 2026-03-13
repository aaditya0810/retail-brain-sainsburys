"""
Retail Brain — WebSockets API
Provides real-time stockout alerts and data ingestion updates to connected dashboards.
"""

from typing import List, Dict
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from logger import get_logger

logger = get_logger("api.websockets")
router = APIRouter(prefix="/api/ws", tags=["Real-time Alerts"])


class ConnectionManager:
    def __init__(self):
        # Maps store_id to a list of active WebSocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, store_id: str):
        await websocket.accept()
        if store_id not in self.active_connections:
            self.active_connections[store_id] = []
        self.active_connections[store_id].append(websocket)
        logger.info(f"WebSocket connected for store {store_id}. Total connections: {len(self.active_connections[store_id])}")

    def disconnect(self, websocket: WebSocket, store_id: str):
        if store_id in self.active_connections:
            self.active_connections[store_id].remove(websocket)
            logger.info(f"WebSocket disconnected for store {store_id}. Total connections: {len(self.active_connections[store_id])}")
            if not self.active_connections[store_id]:
                del self.active_connections[store_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: dict, store_id: str = "ALL"):
        """Broadcast a message to all clients monitoring a specific store, or ALL stores."""
        payload = json.dumps(message)
        
        targets = []
        if store_id == "ALL":
            for conns in self.active_connections.values():
                targets.extend(conns)
        elif store_id in self.active_connections:
            targets = self.active_connections[store_id]
            
        for connection in targets:
            try:
                await connection.send_text(payload)
            except Exception as e:
                logger.error(f"WebSocket broadcast failed: {e}")

manager = ConnectionManager()


@router.websocket("/alerts/{store_id}")
async def websocket_endpoint(websocket: WebSocket, store_id: str):
    """
    Connect to real-time alerts for a specific store.
    Managers can listen to "SBY-LON-001", Admins could listen to "ALL".
    """
    await manager.connect(websocket, store_id)
    try:
        # Send initial confirmation message
        await manager.send_personal_message(
            json.dumps({"type": "system", "message": f"Connected to Retail Brain alerts for store {store_id}"}), 
            websocket
        )
        
        # Keep connection open and handle incoming pings
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await manager.send_personal_message("pong", websocket)
            else:
                 logger.debug(f"Received WS message: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket, store_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, store_id)
