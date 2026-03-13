# 🛒 Retail Brain × Sainsbury's

**End-to-end AI supply chain intelligence — predicting stockouts, forecasting demand, detecting anomalies, and auto-generating purchase orders across 500 Sainsbury's SKUs.**

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)
[![Live on Render](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?logo=render)](https://sainsburys-retail-brain.onrender.com/)

---

## 🚀 Live Demo

**[https://sainsburys-retail-brain.onrender.com/](https://sainsburys-retail-brain.onrender.com/)**

> Free-tier Render — may take ~30 s to wake from sleep on first visit.

---

## 📌 What This Project Does

Retail Brain monitors a simulated Sainsbury's London Flagship store (SBY-LON-001) in real time using five AI models:

| Model | Algorithm | Purpose |
|---|---|---|
| Stockout Predictor | XGBoost (AUC 0.785, 94% recall) | 72-hour stockout probability per SKU |
| Demand Forecaster | Holt-Winters exponential smoothing | 30–90 day demand with UK event uplift |
| Anomaly Detector | IsolationForest + rolling Z-score | Flags supply shocks, promo spikes (20,431 detected) |
| Auto-Replenishment | Rule-based cost-optimised engine | Raises purchase orders automatically |
| Co-Pilot AI | LLM RAG (GPT-4o-mini / rule fallback) | Answers plain-English ops questions |

Dataset: UCI Online Retail II — 1 M+ real UK transactions, cleaned into 500-product daily timeseries.

---

## 🗂️ Project Structure

```
retail-brain-sainsburys/
├── dashboard/          # Streamlit UI (all 9 pages)
│   ├── app.py          # Main application entry point
│   └── charts.py       # Plotly chart helpers
├── src/                # All ML & data modules
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   ├── forecaster.py
│   ├── anomaly_detector.py
│   ├── copilot.py
│   ├── recommendation.py
│   ├── explainer.py
│   └── ...
├── api/                # FastAPI REST backend (optional)
├── models/             # Trained model artefacts (committed)
├── data/raw/           # CSV datasets (committed)
├── data/processed/     # Feature-engineered output
├── scripts/            # Data generation helpers
├── Dockerfile          # For Render Docker deployment
├── render.yaml         # Render service config
└── requirements.txt
```

---

## 💻 Running Locally (any IDE)

### Prerequisites
- Python 3.11+ installed
- Git

### 1 — Clone & install

```bash
git clone https://github.com/aaditya0810/retail-brain-sainsburys.git
cd retail-brain-sainsburys

python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2 — Environment variables (optional)

Copy the example and fill in any keys you have. The app runs fully without them using built-in fallbacks.

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=sk-...          # optional — rule-based Co-Pilot used if absent
OPENWEATHER_API_KEY=...        # optional — synthetic weather used if absent
```

### 3 — Train the model (first run only)

The model artefacts are already committed, so this step is optional. Only needed if you want to retrain from scratch.

```bash
python src/train_model.py
```

### 4 — Launch the dashboard

```bash
# Windows (activate venv first)
venv\Scripts\python.exe -m streamlit run dashboard/app.py

# Mac / Linux
python -m streamlit run dashboard/app.py
```

Opens at **http://localhost:8501**

### IDE-specific tips

| IDE | How to run |
|---|---|
| **VS Code** | Open terminal → run step 4 above. Install the *Python* extension to get venv auto-detection. |
| **PyCharm** | Settings → Python Interpreter → select `venv/Scripts/python.exe` → open terminal → step 4. |
| **Cursor** | Same as VS Code. Cursor auto-detects the venv. |
| **Windsurf** | Open folder → terminal → step 4. |

---

## ☁️ Deploying to Render

The repo contains a `render.yaml` — Render reads it automatically.

### Steps
1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect the GitHub repo `aaditya0810/retail-brain-sainsburys`
3. Render detects `render.yaml` and pre-fills everything
4. Click **Create Web Service**
5. *(Optional)* In **Environment → Secret Files**, add:
   - `OPENAI_API_KEY` — for live GPT-4o-mini Co-Pilot
   - `OPENWEATHER_API_KEY` — for live weather data

Build takes ~3 minutes on first deploy; ~1 minute on subsequent pushes.

---

## 🔑 Environment Variables Reference

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | No | — | GPT-4o-mini Co-Pilot; rule-based fallback used if absent |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI model name |
| `OPENWEATHER_API_KEY` | No | — | Live weather; synthetic fallback used if absent |
| `LOG_LEVEL` | No | `WARNING` | Python logging level |


![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)

## 📌 Project Overview
Retail Brain is an end-to-end ML pipeline and interactive Streamlit dashboard designed to help Sainsbury's store managers identify high-risk product stockouts *before* they occur.

This project was built on a highly realistic synthetic dataset simulating Sainsbury's London Flagship store (SBY-LON-001) during peak UK trading (Q4 2024).

### Key Features
- **Real Sainsbury's Catalogue:** Taste the Difference, Standard, and Branded tiers.
- **UK Market Simulation:** Built-in demand surges for Christmas Eve (+200%), Black Friday, Halloween, and Bank Holidays.
- **Nectar Price Promotions**: Algorithmic promotional uplift across eligible weeks.
- **XGBoost Risk Model:** Predicts 72-hour stockout probability based on 18 engineered features (velocity, stock-to-sales ratio, etc.).
- **LLM Manager Insights:** OpenAI `gpt-4o-mini` generates plain-English briefings explaining *why* a product is at risk.

### Phase 4: Advanced ML & Intelligence
- **Weather & Event Integration:** Pulls weather forecasts (OpenWeatherMap) and local event data (Premier League, Winter Wonderland, etc.) to predict demand spikes for weather-sensitive products like ice cream, soup, and BBQ items.
- **Price Elasticity Model:** Calculates price elasticity of demand per product and category, predicting exactly how much a Nectar price drop will boost sales.
- **Automated Replenishment:** Generates cost-optimised purchase orders with full cost-benefit analysis — holding costs vs. lost revenue — instead of just flagging risk.
- **Reinforcement Learning Agent:** A tile-coded Q-learning decision agent that learns which replenishment strategies maximise profit and minimise waste over time.

---

## 🚀 Live Demo
*(Insert Streamlit Cloud link here after deployment!)*

---

## 🛠️ How to Run Locally

### 1. Clone & Install
```bash
git clone https://github.com/your-username/retail-brain-sainsburys.git
cd retail-brain-sainsburys
pip install -r requirements.txt
```

### 2. Generate Data & Train Model
The repository comes without the large generated CSVs. You need to run the data pipeline once:

```bash
# Generate Q4 2024 Synthetic Dataset (1000 SKUs, ~92,000 daily records)
python scripts/generate_sainsburys_data.py

# Run Feature Engineering (creates rolling averages, velocity trends)
python src/feature_engineering.py

# Train the Classifier (Random Forest + XGBoost)
python src/train_model.py
```

### 3. Launch Dashboard
```bash
streamlit run dashboard/app.py
```
View the dashboard in your browser at `http://localhost:8501`.

### 4. Phase 4: Train Advanced Models (Optional)
```bash
# Train Price Elasticity Model
python src/elasticity.py

# Train RL Decision Agent
python src/rl_agent.py
```

These models enhance the Intelligence Hub and Auto-Orders dashboard pages.
To enable live weather data, add `OPENWEATHER_API_KEY` to your `.env` file (free tier from [openweathermap.org](https://openweathermap.org/api)).

---

## 🧠 Optional: Enable AI Explanations
To enable OpenAI-generated manager explanations on the dashboard:
1. Copy `.env.example` to `.env`
2. Add your OpenAI API Key: `OPENAI_API_KEY=sk-xxxx...`

If no key is provided, the system seamlessly falls back to a professional rule-based explanation engine.

---

*Built as an AI Portfolio Project.*
