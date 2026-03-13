"""
Retail Brain — Database Engine & Session Management
Provides SQLAlchemy engine, session factory, and DB utilities.

Supports PostgreSQL (production) and SQLite (local dev fallback).
Connection string is read from DATABASE_URL environment variable.
"""

import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase

from logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── Database URL ───────────────────────────────────────────────────────────────
# Priority: DATABASE_URL env var > SQLite fallback
_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_SQLITE_PATH = os.path.join(_DB_DIR, "retail_brain.db")
_SQLITE_URL = f"sqlite:///{_SQLITE_PATH}"

DATABASE_URL = os.getenv("DATABASE_URL", _SQLITE_URL)

# Fix for common Heroku/Render postgres:// vs postgresql:// issue
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# ── Engine Configuration ──────────────────────────────────────────────────────
_engine_kwargs = {
    "echo": os.getenv("SQL_ECHO", "false").lower() == "true",
    "pool_pre_ping": True,
}

# SQLite-specific settings
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
    logger.info("Using SQLite database (local dev mode): %s", _SQLITE_PATH)
else:
    _engine_kwargs.update({
        "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
        "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
    })
    logger.info("Using PostgreSQL database")

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ── Base Model ─────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ── Session Helpers ────────────────────────────────────────────────────────────
@contextmanager
def get_session() -> Session:
    """
    Context manager that yields a SQLAlchemy session.
    Automatically commits on success, rolls back on exception.

    Usage:
        with get_session() as session:
            session.add(product)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session():
    """
    FastAPI dependency — yields a DB session for request lifecycle.
    Use with Depends(get_db_session) in route handlers.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db():
    """
    Create all tables defined in ORM models.
    Call this at application startup or in migrations.
    """
    from db_models import Base as ModelsBase  # noqa: avoid circular import
    ModelsBase.metadata.create_all(bind=engine)
    logger.info("Database tables created / verified")


def check_db_connection() -> bool:
    """Test the database connection. Returns True if healthy."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("Database connection failed: %s", e)
        return False
