"""
Retail Brain — Structured Logging
Provides a consistent logging interface across all modules.

Usage:
    from logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone

# ── Configuration ──────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "retail_brain.log")
LOG_FORMAT_ENV = os.getenv("LOG_FORMAT", "text")  # "text" or "json"


# ── JSON Formatter (for production / structured logging) ──────────────────────
class JSONFormatter(logging.Formatter):
    """Outputs structured JSON log lines for production ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


# ── Human-Readable Formatter (for development) ───────────────────────────────
class ColorFormatter(logging.Formatter):
    """Colored console output for developer readability."""

    COLORS = {
        "DEBUG": "\033[90m",       # grey
        "INFO": "\033[36m",        # cyan
        "WARNING": "\033[33m",     # yellow
        "ERROR": "\033[31m",       # red
        "CRITICAL": "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = record.getMessage()
        base = f"{color}{timestamp} [{record.levelname:<8}]{self.RESET} {record.name}: {msg}"
        if record.exc_info and record.exc_info[0] is not None:
            base += f"\n{self.formatException(record.exc_info)}"
        return base


# ── Logger Factory ─────────────────────────────────────────────────────────────
_configured = False


def _configure_root():
    """Configure root logger once — called on first get_logger()."""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("retail_brain")
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Prevent duplicate handlers
    root.handlers.clear()

    # Console handler — always human-readable
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter())
    root.addHandler(console_handler)

    # File handler — JSON in production, text in dev
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    if LOG_FORMAT_ENV == "json":
        file_handler.setFormatter(JSONFormatter())
    else:
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a namespaced logger under the 'retail_brain' hierarchy.

    Args:
        name: Module name, typically __name__

    Returns:
        Configured logging.Logger instance
    """
    _configure_root()
    return logging.getLogger(f"retail_brain.{name}")
