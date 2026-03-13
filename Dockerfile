FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/        ./src/
COPY dashboard/  ./dashboard/
COPY models/     ./models/
COPY data/       ./data/
COPY scripts/    ./scripts/

# Pre-generate engineered features at build time (excluded from git due to size)
RUN python src/feature_engineering.py

# Render injects $PORT at runtime; default 8501 for local Docker
EXPOSE 8501

CMD streamlit run dashboard/app.py \
        --server.port=${PORT:-8501} \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --browser.gatherUsageStats=false
