# 🛒 Retail Brain × Sainsbury's

**AI-powered retail operations intelligence system predicting stockouts across 1000 real Sainsbury's SKUs during Q4 2024.**

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
