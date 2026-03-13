# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY data/ ./data/

# Create logs directory
RUN mkdir -p /app/logs

# Expose ports: 8501 (Streamlit) + 8000 (FastAPI)
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run both services via a startup script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
