"""
Retail Brain — FastAPI Application
REST API for predictions, recommendations, data upload, and health checks.
"""

import os
import sys

# Add src/ to path so bare imports (database, logger, etc.) resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from logger import get_logger
from database import check_db_connection, engine, get_session
from db_models import Base
from api.auth import init_default_admin

# Phase 2 Routers
from api.upload import router as upload_router
from api.auth import router as auth_router
from api.products import router as products_router
from api.predictions import router as predictions_router
from api.websockets import router as ws_router

# Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

logger = get_logger("api.main")
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Retail Brain API",
    description="AI-powered retail stockout prediction and inventory intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Apply rate limiting globally
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow all origins in dev, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Initialize database tables on startup and create default admin."""
    logger.info("Starting Retail Brain API …")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized")
        
        # Start APScheduler
        from api.scheduler import init_scheduler
        init_scheduler()
        
        # Initialize default admin user if DB is fresh
        with get_session() as session:
            init_default_admin(session)
    except Exception as e:
        logger.error("Database initialization failed: %s", e)


@app.on_event("shutdown")
async def shutdown():
    """Stop the scheduler gracefully."""
    from api.scheduler import stop_scheduler
    stop_scheduler()


from api.enterprise import router as enterprise_router
from api.intelligence import router as intelligence_router
from api.forecasting import router as forecasting_router   # Phase 5

# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(products_router)
app.include_router(predictions_router)
app.include_router(ws_router)
app.include_router(enterprise_router)
app.include_router(intelligence_router)
app.include_router(forecasting_router)   # Phase 5: Forecasting + Co-Pilot


@app.get("/health")
@limiter.limit("5/second")
async def health_check(request: Request):
    """Health check endpoint for monitoring and load balancers."""
    db_ok = check_db_connection()
    return {
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected",
        "version": "1.0.0",
    }


@app.get("/")
async def root():
    """API root — basic info and links."""
    return {
        "name": "Retail Brain API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "upload_products": "POST /api/upload/products",
            "upload_sales": "POST /api/upload/sales",
            "upload_inventory": "POST /api/upload/inventory",
            "upload_status": "GET /api/upload/status",
        },
    }


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
