#!/bin/bash
set -e

echo "🧠 Retail Brain — Starting services …"

# Start FastAPI in background
echo "🚀 Starting FastAPI API server on port 8000 …"
cd /app
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit
echo "🚀 Starting Streamlit dashboard on port 8501 …"
cd /app
streamlit run dashboard/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
