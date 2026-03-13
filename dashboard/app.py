"""
Retail Brain × Sainsbury's — Stockout Intelligence Dashboard
AI-powered retail operations for Sainsbury's Q4 2024 (Store: SBY-LON-001)

Run: streamlit run dashboard/app.py
"""

import os
import sys
import time
import pandas as pd
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "dashboard"))

from data_ingestion   import build_base_dataset, load_sales, load_inventory, load_products
from feature_engineering import compute_features
from predict          import load_model, run_inference
from recommendation   import generate_recommendations
from explainer        import generate_explanation
from charts import (
    risk_bar_chart, risk_distribution_pie,
    sales_trend_chart, stock_trend_chart, risk_gauge
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Brain × Sainsbury's",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — Retail Brain v3.0 Enterprise Light UI ──────────────────────────────
st.markdown("""
<style>
/* =================================================================
   Retail Brain x Sainsbury's — Enterprise Light UI v3.0
   Primary: Sainsbury's Orange #F06A00
   Style: Clean enterprise (Atlassian / Power BI / Tableau aesthetic)
   ================================================================= */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  -webkit-font-smoothing: antialiased;
}

/* ── App background ── */
.stApp { background: #f4f5f7; color: #172b4d; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #ffffff !important;
  border-right: 1px solid #dfe1e6 !important;
  box-shadow: 2px 0 8px rgba(9,30,66,.06) !important;
}
[data-testid="stSidebar"] * { color: #172b4d !important; }
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* Brand header strip */
.brand-header {
  padding: 20px 20px 16px;
  background: linear-gradient(135deg, #c4440a 0%, #F06A00 100%);
  position: relative; overflow: hidden;
}
.brand-header::after {
  content: ''; position: absolute; top: -40px; right: -40px;
  width: 110px; height: 110px; background: rgba(255,255,255,.08); border-radius: 50%;
}
.brand-name {
  font-size: 1.1rem; font-weight: 800; letter-spacing: -.4px;
  color: #fff; margin: 0 0 3px; position: relative; z-index: 1;
}
.brand-tagline {
  font-size: .67rem; color: rgba(255,255,255,.88);
  line-height: 1.45; position: relative; z-index: 1;
}

/* Live status dot */
@keyframes pulse-green {
  0%, 100% { box-shadow: 0 0 0 0 rgba(0,135,90,.5); }
  50%       { box-shadow: 0 0 0 5px rgba(0,135,90,0); }
}
.status-dot {
  display: inline-block; width: 7px; height: 7px;
  background: #00875a; border-radius: 50%;
  animation: pulse-green 2.2s infinite;
  margin-right: 5px; vertical-align: middle;
}

/* Sidebar stat box */
.sb-stat {
  background: #fff7f0; border: 1px solid #fed7aa;
  border-radius: 8px; padding: 10px 14px; margin-bottom: 6px;
}
.sb-stat-val { font-size: 1.35rem; font-weight: 800; color: #de350b !important; line-height: 1; }
.sb-stat-lbl { font-size: .67rem; color: #5e6c84 !important; margin-top: 3px; }

/* ── KPI Cards ── */
.kpi-card {
  background: #ffffff;
  border: 1px solid #dfe1e6;
  border-top: 3px solid #F06A00;
  border-radius: 8px; padding: 20px 16px;
  text-align: center;
  box-shadow: 0 1px 3px rgba(9,30,66,.06), 0 1px 2px rgba(9,30,66,.04);
  transition: all .2s ease;
  position: relative;
}
.kpi-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 6px 20px rgba(9,30,66,.10);
  border-top-color: #c4440a;
}
.kpi-value {
  font-size: 2.15rem; font-weight: 800; line-height: 1; letter-spacing: -1.5px;
  color: #172b4d;
}
.kpi-label {
  font-size: .72rem; color: #5e6c84; margin-top: 7px;
  font-weight: 600; text-transform: uppercase; letter-spacing: .06em;
}
.kpi-sub { font-size: .67rem; color: #97a0af; margin-top: 3px; }

/* ── Feature / Tech cards ── */
.feature-card {
  background: #ffffff;
  border: 1px solid #dfe1e6;
  border-radius: 8px; padding: 22px 18px;
  box-shadow: 0 1px 3px rgba(9,30,66,.05);
  transition: all .2s ease;
  height: 100%;
}
.feature-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 6px 18px rgba(9,30,66,.09);
  border-color: #F06A00;
}
.feature-icon { font-size: 1.8rem; margin-bottom: 10px; display: block; }
.feature-title { font-size: .95rem; font-weight: 700; color: #172b4d; margin-bottom: 7px; }
.feature-desc { font-size: .76rem; color: #5e6c84; line-height: 1.58; }
.feature-tag {
  display: inline-block; margin-top: 12px;
  background: #fff7ed; color: #c4440a;
  border: 1px solid #fed7aa;
  font-size: .62rem; font-weight: 800;
  padding: 2px 9px; border-radius: 4px;
  text-transform: uppercase; letter-spacing: .08em;
}

/* ── Pipeline steps ── */
.pipeline-step {
  background: #ffffff;
  border: 1px solid #dfe1e6;
  border-radius: 8px; padding: 20px 16px; text-align: center;
  box-shadow: 0 1px 3px rgba(9,30,66,.05);
  transition: all .2s; height: 100%;
}
.pipeline-step:hover { border-color: #F06A00; transform: translateY(-2px); box-shadow: 0 6px 16px rgba(9,30,66,.08); }
.pipeline-num {
  display: inline-flex; align-items: center; justify-content: center;
  width: 30px; height: 30px;
  background: #F06A00;
  border-radius: 50%; font-size: .78rem; font-weight: 900; color: #fff;
  margin-bottom: 12px;
}
.pipeline-title { font-size: .9rem; font-weight: 700; color: #172b4d; margin-bottom: 7px; }
.pipeline-desc { font-size: .73rem; color: #5e6c84; line-height: 1.52; }

/* ── Hero banner ── */
.hero-banner {
  position: relative;
  background: #ffffff;
  border: 1px solid #dfe1e6;
  border-left: 5px solid #F06A00;
  border-radius: 8px; padding: 36px 44px; margin-bottom: 28px;
  box-shadow: 0 2px 6px rgba(9,30,66,.07);
}
.hero-title {
  font-size: 2.8rem; font-weight: 900; line-height: 1.05;
  letter-spacing: -1.8px; margin: 0 0 8px;
  color: #F06A00;
}
.hero-subtitle {
  font-size: 1.15rem; color: #5e6c84; font-weight: 400;
  letter-spacing: -.2px; margin-bottom: 14px;
}
.hero-sub {
  font-size: .96rem; color: #42526e; max-width: 720px;
  line-height: 1.68; margin: 0 0 20px;
}
.hero-chip {
  display: inline-flex; align-items: center; gap: 4px;
  background: #f4f5f7; border: 1px solid #dfe1e6;
  padding: 4px 10px; border-radius: 4px;
  font-size: .72rem; color: #42526e; font-weight: 500;
  margin-right: 6px; margin-bottom: 5px;
}

/* ── Section headers ── */
.section-header {
  display: flex; align-items: center; gap: 10px;
  font-size: .95rem; font-weight: 700; color: #172b4d;
  margin: 28px 0 14px;
}
.section-header::after {
  content: ''; flex: 1; height: 1px; background: #dfe1e6;
}

/* ── Insight cards ── */
.insight-card {
  background: #ffffff; border: 1px solid #dfe1e6;
  border-left: 4px solid #F06A00; border-radius: 8px;
  padding: 15px 18px; margin-bottom: 12px;
  line-height: 1.65; color: #172b4d;
  box-shadow: 0 1px 3px rgba(9,30,66,.04);
  transition: border-left-color .2s, transform .2s;
}
.insight-card:hover { border-left-color: #7B2D8B; transform: translateX(2px); }
.insight-product { font-size: .95rem; font-weight: 700; color: #F06A00; margin-bottom: 7px; }

/* ── Risk / tier badges ── */
.badge-ttd     { background:#f3e8ff; color:#6d28d9; border:1px solid #c4b5fd; padding:2px 10px; border-radius:4px; font-size:.72rem; font-weight:700; }
.badge-std     { background:#fff7ed; color:#c4440a; border:1px solid #fed7aa; padding:2px 10px; border-radius:4px; font-size:.72rem; font-weight:700; }
.badge-branded { background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; padding:2px 10px; border-radius:4px; font-size:.72rem; font-weight:700; }
.badge-danger  { background:#ffebe6; color:#de350b; border:1px solid #ffbdad; padding:2px 10px; border-radius:4px; font-size:.72rem; font-weight:700; }
.badge-warning { background:#fffae6; color:#974f0c; border:1px solid #ffe380; padding:2px 10px; border-radius:4px; font-size:.72rem; font-weight:700; }
.badge-success { background:#e3fcef; color:#006644; border:1px solid #abf5d1; padding:2px 10px; border-radius:4px; font-size:.72rem; font-weight:700; }

/* ── Event pill ── */
.event-pill {
  display: inline-flex; align-items: center; gap: 6px;
  background: #fff7ed; border: 1px solid #fed7aa;
  color: #c4440a; padding: 5px 16px; border-radius: 5px;
  font-size: .78rem; font-weight: 600;
}

/* ── Buttons ── */
.stButton > button {
  background: #F06A00 !important;
  color: #fff !important; border: none !important;
  border-radius: 6px !important; font-weight: 600 !important;
  box-shadow: none !important;
  transition: background .18s !important;
}
.stButton > button:hover {
  background: #c4440a !important;
  box-shadow: 0 2px 8px rgba(196,68,10,.28) !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
  background: #ffffff !important;
  border: 1px solid #dfe1e6 !important;
  border-radius: 8px !important; padding: 16px !important;
  box-shadow: 0 1px 3px rgba(9,30,66,.05) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #F06A00 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: #ffffff !important;
  border: 1px solid #dfe1e6 !important; border-radius: 8px !important;
}

/* ── Selectbox ── */
[data-baseweb="select"] > div { background: #ffffff !important; border-color: #dfe1e6 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #ffffff !important; border-radius: 6px !important; gap: 2px !important; padding: 3px !important; border: 1px solid #dfe1e6 !important; }
.stTabs [data-baseweb="tab"] { border-radius: 5px !important; color: #5e6c84 !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { background: #fff7ed !important; color: #c4440a !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; border: 1px solid #dfe1e6; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f4f5f7; }
::-webkit-scrollbar-thumb { background: #c1c7d0; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #F06A00; }

/* ── Misc ── */
hr { border-color: #dfe1e6 !important; }
.stSpinner > div { border-top-color: #F06A00 !important; }
.stProgress > div > div { background: #F06A00 !important; }
@keyframes risk-blink { 0%,100%{opacity:1;} 50%{opacity:.65;} }
.risk-blink { animation: risk-blink 2s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    return load_model()

@st.cache_data(show_spinner=False)
def get_predictions():
    model, meta = load_resources()
    preds = run_inference(model, meta)
    return generate_recommendations(preds)

@st.cache_data(show_spinner=False)
def get_raw_sales():     return load_sales()
@st.cache_data(show_spinner=False)
def get_raw_inventory(): return load_inventory()
@st.cache_data(show_spinner=False)
def get_products():      return load_products()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Brand Header
    st.markdown("""
    <div class="brand-header">
      <div class="brand-name">🛒 Retail Brain</div>
      <div class="brand-tagline">
        Sainsbury's AI Stockout Intelligence<br>
        Store SBY-LON-001 · London Flagship
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Live status
    st.markdown(
        "<div style='padding:9px 16px 2px;'>"
        "<span class='status-dot'></span>"
        "<span style='font-size:.73rem;color:#00875a;font-weight:700;'>LIVE</span>"
        "<span style='font-size:.7rem;color:#5e6c84;margin-left:7px;'>All systems operational</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Navigation
    st.markdown(
        "<p style='font-size:.6rem;font-weight:700;color:#5e6c84;letter-spacing:.12em;"
        "text-transform:uppercase;margin:2px 0 4px;padding:0 6px;'>Navigate</p>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "nav",
        ["🏠 Overview", "📋 Risk Table", "🔍 Product Detail",
         "💡 Manager Insights", "📅 Event Calendar",
         "🧠 Intelligence Hub", "📦 Auto-Orders",
         "📈 Demand Forecast", "🤖 Co-Pilot"],
        label_visibility="collapsed",
    )
    st.divider()

    # ── Quick stat: high-risk SKU count
    try:
        _df = get_predictions()
        _hi = int((_df["risk_level"] == "High").sum())
        _tot = len(_df)
        st.markdown(
            f"<div class='sb-stat'>"
            f"<div class='sb-stat-val'>{_hi}</div>"
            f"<div class='sb-stat-lbl'>High-risk SKUs right now (of {_tot})</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    # ── Project phase guide
    st.markdown(
        "<p style='font-size:.6rem;font-weight:700;color:#5e6c84;letter-spacing:.12em;"
        "text-transform:uppercase;margin:10px 0 6px;padding:0 6px;'>Project Phases</p>",
        unsafe_allow_html=True,
    )
    _phases = [
        ("1", "Data Ingestion",   "#4338ca", "#eef2ff",  "#c7d2fe"),
        ("2", "ML Predictions",   "#c4440a", "#fff7ed",  "#fed7aa"),
        ("3", "Explainability",   "#047857", "#ecfdf5",  "#a7f3d0"),
        ("4", "Intelligence Hub", "#7c3aed", "#f5f3ff",  "#ddd6fe"),
        ("5", "Co-Pilot AI",      "#c4440a", "#fff7ed",  "#fed7aa"),
    ]
    _ph_html = "<div style='display:flex;flex-direction:column;gap:4px;padding:0 2px;'>"
    for _n, _lbl, _c, _bg, _bd in _phases:
        _ph_html += (
            f"<div style='display:flex;align-items:center;gap:8px;background:{_bg};"
            f"border:1px solid {_bd};border-radius:8px;padding:6px 10px;'>"
            f"<span style='display:inline-flex;align-items:center;justify-content:center;"
            f"min-width:18px;height:18px;background:{_c}22;border-radius:4px;"
            f"font-size:.65rem;font-weight:800;color:{_c};'>{_n}</span>"
            f"<span style='font-size:.72rem;color:{_c};font-weight:600;'>{_lbl}</span>"
            f"</div>"
        )
    _ph_html += "</div>"
    st.markdown(_ph_html, unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown(
        "<p style='color:#97a0af;font-size:.65rem;margin-top:10px;text-align:center;line-height:1.6;'>"
        "Retail Brain v2.0 · Sainsbury's PLC<br>"
        "XGBoost · Holt-Winters · IsolationForest · RAG</p>",
        unsafe_allow_html=True,
    )


# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading Sainsbury's intelligence engine …"):
    try:
        df = get_predictions()
    except FileNotFoundError:
        st.error(
            "⚠️ Model not found. Please run the setup pipeline:\n\n"
            "```bash\n"
            "python scripts/generate_sainsburys_data.py\n"
            "python src/feature_engineering.py\n"
            "python src/train_model.py\n"
            "```"
        )
        st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    prods_df = get_products()

    # ── Hero Banner ────────────────────────────────────────────────────────────
    # Active event detection (shown in hero chips)
    _event_chip = ""
    try:
        _sales_tmp = get_raw_sales()
        _ld = pd.to_datetime(_sales_tmp["date"]).max()
        _ev = _sales_tmp[_sales_tmp["date"] == _ld]["uk_event"].mode()[0]
        if _ev != "Normal":
            _event_chip = f"<span class='hero-chip' style='border-color:rgba(240,106,0,.4);color:#fb923c;'>📅 {_ev}</span>"
    except Exception:
        pass

    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-title">Retail Brain</div>
      <div class="hero-subtitle">AI-Powered Stockout Intelligence &amp; Demand Forecasting</div>
      <p class="hero-sub">
        A portfolio-grade supply chain intelligence system modelling a fictional
        <strong style="color:#F06A00;">Sainsbury's London flagship (SBY-LON-001)</strong>.
        Built on a synthetic dataset of <strong style="color:#F06A00;">500 real Sainsbury's SKUs</strong>
        with authentic UK seasonal demand patterns (Q4 2024 · Oct–Dec · Halloween, Bonfire Night,
        Christmas). Five ML models work together to predict stockouts 7 days ahead, forecast
        demand up to 90 days, detect anomalies, and auto-generate purchase orders.
      </p>
      <div>
        <span class="hero-chip">🏪 Store SBY-LON-001 (Fictional)</span>
        <span class="hero-chip">📦 500 Sainsbury's SKUs</span>
        <span class="hero-chip">📅 Q4 2024 — Oct to Dec</span>
        <span class="hero-chip">🤖 5 ML Models</span>
        <span class="hero-chip">🧪 Synthetic Dataset</span>
        {_event_chip}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── How It Works — 4-step pipeline ────────────────────────────────────────
    st.markdown('<div class="section-header">🔬 How It Works</div>', unsafe_allow_html=True)
    _p1, _p2, _p3, _p4 = st.columns(4)
    for _col, _n, _title, _desc in zip(
        [_p1, _p2, _p3, _p4],
        ["1", "2", "3", "4"],
        ["Ingest", "Predict", "Forecast", "Act"],
        [
            "Synthetic Q4 2024 dataset: 500 real Sainsbury's SKUs across Dairy, Bakery, Meat, Produce &amp; more. Daily sales rows per product, generated with realistic UK seasonal patterns (Halloween, Christmas, Bonfire Night).",
            "XGBoost scores each product's 7-day stockout probability using 14 engineered features (AUC&nbsp;0.785, 94%&nbsp;recall).",
            "Holt-Winters exponential smoothing generates 30–90-day demand forecasts with seasonality &amp; UK event uplift.",
            "Auto-PO engine raises replenishment orders · IsolationForest flags anomalies · LLM Co-Pilot answers ops questions.",
        ],
    ):
        _col.markdown(
            f'<div class="pipeline-step">'
            f'<div class="pipeline-num">{_n}</div>'
            f'<div class="pipeline-title">{_title}</div>'
            f'<div class="pipeline-desc">{_desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AI Model Stack ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ AI Model Stack</div>', unsafe_allow_html=True)
    _t1, _t2, _t3, _t4 = st.columns(4)
    for _col, _icon, _title, _desc, _tag in zip(
        [_t1, _t2, _t3, _t4],
        ["🎯", "📈", "🔍", "🤖"],
        ["Stockout Predictor", "Demand Forecaster", "Anomaly Detector", "Co-Pilot AI"],
        [
            "XGBoost gradient-boosted trees — 14 features, AUC 0.785, 94% recall on stockout class. Scores all 500 products daily.",
            "Holt-Winters double exponential smoothing with category seasonality &amp; UK holiday uplift. Horizons up to 90 days with 90% CI.",
            "IsolationForest + rolling Z-score. Detected 20,431 anomalies across 2 years with root-cause tags: supply chain, promo, shock.",
            "RAG-based assistant using GPT-4o-mini (or rule-based fallback). Answers any ops question with live risk, forecast &amp; anomaly context.",
        ],
        ["XGBoost · Phase 2", "Holt-Winters · Phase 5", "IsolationForest · Phase 5", "LLM RAG · Phase 5"],
    ):
        _col.markdown(
            f'<div class="feature-card">'
            f'<div class="feature-icon">{_icon}</div>'
            f'<div class="feature-title">{_title}</div>'
            f'<div class="feature-desc">{_desc}</div>'
            f'<div class="feature-tag">{_tag}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dataset info box ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#fffae6;border:1px solid #ffe380;border-left:4px solid #F06A00;
                border-radius:8px;padding:14px 20px;margin-bottom:8px;">
      <span style="font-size:.82rem;font-weight:700;color:#974f0c;">📋 About This Dataset</span>
      <span style="font-size:.79rem;color:#5e6c84;margin-left:12px;">
        <strong>Synthetic demo data</strong> — 500 real Sainsbury's SKU names &amp; prices ·
        Q4 2024 (Oct 1 – Dec 31) · Daily sales per product with UK seasonal uplifts (Halloween,
        Bonfire Night, Christmas) · Simulated stockouts, promotions &amp; lead times.
        All predictions &amp; forecasts below are generated by ML models trained on this synthetic dataset.
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Live KPIs ──────────────────────────────────────────────────────────────
    total    = len(df)
    n_high   = int((df["risk_level"] == "High").sum())
    n_medium = int((df["risk_level"] == "Medium").sum())
    avg_doc  = df["days_of_cover"].clip(upper=999).mean()
    avg_risk = df["stockout_probability"].mean() * 100
    n_ttd    = int((prods_df["tier"] == "Taste the Difference").sum())

    st.markdown('<div class="section-header">📊 Live System Statistics</div>', unsafe_allow_html=True)
    _c1, _c2, _c3, _c4, _c5, _c6 = st.columns(6)
    for _col, _val, _lbl, _sub, _color in [
        (_c1, str(total),          "Total SKUs",      "Products monitored",      ""),
        (_c2, str(n_high),         "High Risk",        "Replenish immediately",   "#f87171"),
        (_c3, str(n_medium),       "Medium Risk",      "Monitor closely",         "#fbbf24"),
        (_c4, f"{avg_doc:.1f}d",   "Days of Cover",    "Average across portfolio", ""),
        (_c5, f"{avg_risk:.0f}%",  "Avg Risk Score",   "Portfolio-wide",          ""),
        (_c6, str(n_ttd),          "Premium SKUs",     "Taste the Difference",    "#c084fc"),
    ]:
        _cs = f'style="-webkit-text-fill-color:{_color};"' if _color else ""
        _col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-value" {_cs}>{_val}</div>'
            f'<div class="kpi-label">{_lbl}</div>'
            f'<div class="kpi-sub">{_sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ─────────────────────────────────────────────────────────────
    col_bar, col_pie = st.columns([2, 1])
    with col_bar:
        st.plotly_chart(risk_bar_chart(df, n=15), use_container_width=True)
    with col_pie:
        st.plotly_chart(risk_distribution_pie(df), use_container_width=True)

    # ── Tier breakdown (KPI card style) ───────────────────────────────────────
    st.markdown('<div class="section-header">🏷️ Stockout Risk by Product Tier</div>', unsafe_allow_html=True)
    _merged = df.merge(prods_df[["product_id", "tier"]], on="product_id", how="left")
    _tier_data = _merged.groupby("tier")["stockout_probability"].agg(["mean", "count"]).reset_index()
    _tier_data.columns = ["Tier", "AvgRisk", "Products"]
    _tier_cols = st.columns(len(_tier_data))
    _tier_palette = {"Taste the Difference": "#c084fc", "Sainsbury's": "#fb923c", "Branded": "#818cf8"}
    for _tcol, (_, _tr) in zip(_tier_cols, _tier_data.iterrows()):
        _tc = _tier_palette.get(_tr["Tier"], "#8b949e")
        _tcol.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-value" style="-webkit-text-fill-color:{_tc};font-size:1.8rem;">'
            f'{_tr["AvgRisk"]*100:.1f}%</div>'
            f'<div class="kpi-label">{_tr["Tier"]}</div>'
            f'<div class="kpi-sub">{int(_tr["Products"])} products</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Immediate replenishment alerts ─────────────────────────────────────────
    high_risk = df[df["risk_level"] == "High"].sort_values("stockout_probability", ascending=False)
    if not high_risk.empty:
        st.markdown('<div class="section-header">🚨 Immediate Replenishment Required</div>',
                    unsafe_allow_html=True)
        # Column header
        _h0, _h1, _h2, _h3, _h4 = st.columns([3.5, 1.2, 1.2, 1.5, 2])
        for _hcol, _htxt in zip([_h0, _h1, _h2, _h3, _h4],
                                 ["Product", "Risk", "Time Left", "Order Qty", "Revenue at Risk"]):
            _hcol.markdown(f"<span style='font-size:.7rem;font-weight:700;color:#5e6c84;text-transform:uppercase;letter-spacing:.06em;'>{_htxt}</span>",
                           unsafe_allow_html=True)
        st.markdown("<hr style='margin:4px 0 8px;border-color:rgba(255,255,255,.06);'>", unsafe_allow_html=True)
        for _, row in high_risk.head(8).iterrows():
            ca, cb, cc, cd, ce = st.columns([3.5, 1.2, 1.2, 1.5, 2])
            ca.markdown(f"<span style='font-size:.88rem;font-weight:600;color:#172b4d;'>{row['product_name']}</span>",
                        unsafe_allow_html=True)
            cb.markdown(f"<span class='badge-danger risk-blink'>{row['stockout_probability']*100:.0f}%</span>",
                        unsafe_allow_html=True)
            cc.markdown(f"<span style='font-size:.83rem;color:#5e6c84;'>{row['time_to_stockout']}</span>",
                        unsafe_allow_html=True)
            cd.markdown(f"<span style='font-size:.85rem;font-weight:600;color:#c4440a;'>{row['replenishment_qty']:,} units</span>",
                        unsafe_allow_html=True)
            price_row = prods_df[prods_df["product_id"] == row.get("product_id", "")]
            if not price_row.empty:
                rev_risk = round(row["replenishment_qty"] * float(price_row.iloc[0]["unit_price"]), 2)
                ce.markdown(f"<span style='font-size:.85rem;font-weight:700;color:#de350b;'>£{rev_risk:,.2f}</span>",
                            unsafe_allow_html=True)
            st.markdown("<hr style='margin:4px 0;border-color:rgba(0,0,0,.07);'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK TABLE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Risk Table":
    st.markdown("# 📋 Product Risk Table")
    st.markdown("<p style='color:#8b949e;'>All Sainsbury's SKUs ranked by AI stockout probability.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    prods_df = get_products()
    merged   = df.merge(prods_df[["product_id","tier","unit_price"]], on="product_id", how="left")

    f1, f2, f3, f4 = st.columns([2, 2, 1.5, 1.5])
    risk_filter = f1.multiselect("Risk Level", ["High","Medium","Low"],
                                  default=["High","Medium","Low"])
    cat_filter  = f2.multiselect("Category",
                                  sorted(prods_df["category"].unique()),
                                  default=list(prods_df["category"].unique()))
    tier_filter = f3.multiselect("Tier",
                                  sorted(prods_df["tier"].unique()),
                                  default=list(prods_df["tier"].unique()))
    sort_by     = f4.selectbox("Sort By", ["Stockout Risk ↓","Days of Cover ↑","Revenue at Risk ↓"])

    filtered = merged[
        merged["risk_level"].isin(risk_filter) &
        merged["category"].isin(cat_filter) &
        merged["tier"].isin(tier_filter)
    ].copy()

    filtered["revenue_at_risk"] = (
        filtered["replenishment_qty"] * filtered["unit_price"]
    ).round(2)

    if sort_by == "Stockout Risk ↓":
        filtered = filtered.sort_values("stockout_probability", ascending=False)
    elif sort_by == "Days of Cover ↑":
        filtered = filtered.sort_values("days_of_cover", ascending=True)
    else:
        filtered = filtered.sort_values("revenue_at_risk", ascending=False)

    table = filtered[[
        "risk_emoji","product_name","tier","category",
        "stock_on_hand","days_of_cover","sales_velocity_7d",
        "stockout_probability","time_to_stockout",
        "replenishment_qty","revenue_at_risk",
    ]].copy()
    table.columns = ["⚠️","Product","Tier","Category","Stock",
                     "Days Cover","Velocity 7d","Risk %",
                     "Time to Stockout","Order Qty","Rev. at Risk £"]
    table["Risk %"] = (table["Risk %"] * 100).round(1).astype(str) + "%"
    table["Days Cover"] = table["Days Cover"].round(1)
    table["Velocity 7d"] = table["Velocity 7d"].round(1)
    table["Stock"] = table["Stock"].round(0).astype(int)

    st.dataframe(table.reset_index(drop=True), use_container_width=True, height=540)
    st.markdown(f"<p style='color:#484f58;font-size:0.78rem;'>Showing {len(filtered)} of {len(df)} SKUs</p>",
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PRODUCT DETAIL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Product Detail":
    st.markdown("# 🔍 Product Detail")
    st.markdown("<p style='color:#8b949e;'>Deep-dive analytics for a single Sainsbury's SKU.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    prods_df = get_products()
    sorted_df = df.sort_values("stockout_probability", ascending=False)
    options  = sorted_df.apply(
        lambda r: f"{r['risk_emoji']} {r['product_name']} — {r['stockout_probability']*100:.0f}%", axis=1
    ).tolist()
    ids = sorted_df["product_id"].tolist()

    sel_label = st.selectbox("Select SKU", options)
    sel_idx   = options.index(sel_label)
    sel_id    = ids[sel_idx]
    sel_row   = sorted_df[sorted_df["product_id"] == sel_id].iloc[0]
    prod_meta = prods_df[prods_df["product_id"] == sel_id].iloc[0]

    # Tier badge
    tier_badge = {
        "Taste the Difference": '<span class="badge-ttd">Taste the Difference</span>',
        "Sainsbury's":          '<span class="badge-std">Sainsbury\'s</span>',
        "Branded":              '<span class="badge-branded">Branded</span>',
    }.get(prod_meta["tier"], "")

    st.markdown(f"<p>{tier_badge} &nbsp; <span style='color:#8b949e;font-size:0.85rem;'>SKU: {sel_id} · £{prod_meta['unit_price']}</span></p>",
                unsafe_allow_html=True)

    g_col, m1, m2, m3, m4 = st.columns([2, 1, 1, 1, 1])
    with g_col:
        st.plotly_chart(risk_gauge(sel_row["stockout_probability"], sel_row["product_name"]),
                        use_container_width=True)
    m1.metric("Category",    prod_meta["category"])
    m2.metric("Stock on Hand", f"{sel_row['stock_on_hand']:.0f} units")
    m3.metric("Days of Cover", f"{sel_row['days_of_cover']:.1f}")
    m4.metric("Time to Stockout", sel_row["time_to_stockout"])

    st.markdown("---")
    risk   = sel_row["risk_level"]
    badge  = {"High":"danger","Medium":"warning","Low":"success"}[risk]
    promo_flag = "🏷️ Nectar Price Active" if sel_row.get("promo_days_last_7", 0) > 0 else ""
    st.markdown(
        f"""<div class="insight-card">
            <div class="insight-product">{sel_row['risk_emoji']} {sel_row['product_name']}</div>
            <span class="badge-{badge}">{risk} Risk — {sel_row['stockout_probability']*100:.1f}%</span>
            &nbsp;&nbsp;{promo_flag}
            &nbsp;|&nbsp; Action: <b>{sel_row['recommended_action']}</b>
            {f" &nbsp;|&nbsp; Order <b>{sel_row['replenishment_qty']} units</b> (£{sel_row['replenishment_qty']*prod_meta['unit_price']:.2f})" if sel_row['replenishment_qty'] > 0 else ""}
        </div>""",
        unsafe_allow_html=True,
    )

    sales_df = get_raw_sales()
    inv_df   = get_raw_inventory()
    cl, cr = st.columns(2)
    with cl:
        st.plotly_chart(
            sales_trend_chart(sales_df, sel_id, sel_row["product_name"]),
            use_container_width=True,
        )
    with cr:
        st.plotly_chart(
            stock_trend_chart(inv_df, sel_id, sel_row["product_name"],
                              float(prod_meta["reorder_point"])),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MANAGER INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Manager Insights":
    st.markdown("# 💡 Manager Insights")
    st.markdown(
        "<p style='color:#8b949e;'>AI-generated plain-English briefings for store managers.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    high_medium = df[df["risk_level"].isin(["High","Medium"])].sort_values(
        "stockout_probability", ascending=False
    ).head(10)

    if high_medium.empty:
        st.success("✅ Excellent! No high or medium risk products at this time.")
    else:
        n_cols = st.slider("Products to analyse", 3, 10, 6)
        top_n  = high_medium.head(n_cols)
        btn    = st.button("✨ Generate AI Manager Briefings")

        if btn or "sby_explanations" in st.session_state:
            if btn:
                expls = {}
                prog  = st.progress(0, text="Generating Sainsbury's briefings …")
                for i, (_, row) in enumerate(top_n.iterrows()):
                    expls[row["product_id"]] = generate_explanation(row.to_dict())
                    prog.progress((i+1)/len(top_n), text=f"Analysing {row['product_name']} …")
                    time.sleep(0.05)
                prog.empty()
                st.session_state["sby_explanations"] = expls
            expls = st.session_state.get("sby_explanations", {})
            prods_df = get_products()

            for _, row in top_n.iterrows():
                pid    = row["product_id"]
                badge  = {"High":"danger","Medium":"warning","Low":"success"}[row["risk_level"]]
                meta   = prods_df[prods_df["product_id"] == pid]
                price  = float(meta.iloc[0]["unit_price"]) if not meta.empty else 0
                rev    = row["replenishment_qty"] * price

                st.markdown(
                    f"""<div class="insight-card">
                        <div class="insight-product">
                          {row['risk_emoji']} {row['product_name']}
                          &nbsp;<span class="badge-{badge}">{row['risk_level']} — {row['stockout_probability']*100:.0f}%</span>
                        </div>
                        <p style="margin:6px 0 10px;">{expls.get(pid, '_Generating …_')}</p>
                        <span style="color:#484f58;font-size:0.78rem;">
                          📦 Stock: {row['stock_on_hand']:.0f} units &nbsp;·&nbsp;
                          ⏱️ {row['time_to_stockout']} &nbsp;·&nbsp;
                          🛒 Order {row['replenishment_qty']} units &nbsp;·&nbsp;
                          💷 Revenue at risk: £{rev:.2f}
                        </span>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("👆 Click **Generate AI Manager Briefings** to get insights.")
            prods_df = get_products()
            for _, row in top_n.iterrows():
                badge = {"High":"danger","Medium":"warning","Low":"success"}[row["risk_level"]]
                meta  = prods_df[prods_df["product_id"] == row["product_id"]]
                price = float(meta.iloc[0]["unit_price"]) if not meta.empty else 0
                rev   = row["replenishment_qty"] * price
                st.markdown(
                    f"""<div class="insight-card">
                        <div class="insight-product">
                          {row['risk_emoji']} {row['product_name']}
                          &nbsp;<span class="badge-{badge}">{row['risk_level']} — {row['stockout_probability']*100:.0f}%</span>
                        </div>
                        <span style="color:#484f58;font-size:0.78rem;">
                          📦 {row['stock_on_hand']:.0f} units &nbsp;·&nbsp;
                          ⏱️ {row['time_to_stockout']} &nbsp;·&nbsp;
                          🛒 Order {row['replenishment_qty']} &nbsp;·&nbsp;
                          💷 £{rev:.2f} at risk
                        </span>
                    </div>""",
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EVENT CALENDAR (Sainsbury's-specific bonus page)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📅 Event Calendar":
    st.markdown("# 📅 Q4 2024 UK Event Impact")
    st.markdown(
        "<p style='color:#8b949e;'>How UK shopping events affect Sainsbury's sales velocity and stockout risk.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    try:
        from data_ingestion import load_calendar
        cal_df = load_calendar()
        sales_df = get_raw_sales()

        # Merge sales totals by date with calendar
        daily_totals = sales_df.groupby("date")["units_sold"].sum().reset_index()
        daily_totals.rename(columns={"units_sold":"total_units"}, inplace=True)
        daily_totals["date"] = pd.to_datetime(daily_totals["date"])
        merged = daily_totals.merge(cal_df[["date","uk_event","event_multiplier","is_nectar_week"]], on="date")

        # Event summary table
        event_summary = (
            merged.groupby("uk_event")
            .agg(
                avg_daily_units=("total_units","mean"),
                peak_units=("total_units","max"),
                days=("total_units","count"),
            )
            .reset_index()
            .sort_values("avg_daily_units", ascending=False)
        )
        event_summary["avg_daily_units"]  = event_summary["avg_daily_units"].round(0).astype(int)
        event_summary["peak_units"]       = event_summary["peak_units"].round(0).astype(int)
        event_summary.columns = ["UK Event","Avg Daily Units","Peak Daily Units","Trading Days"]

        st.markdown('<div class="section-header">🗓️ Sales Impact by UK Shopping Event</div>',
                    unsafe_allow_html=True)
        st.dataframe(event_summary, use_container_width=True, height=320)

        # Daily sales line chart
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=merged["date"], y=merged["total_units"],
            mode="lines", name="Daily Store Sales",
            line=dict(color="#F06A00", width=2),
            fill="tozeroy", fillcolor="rgba(240,106,0,0.08)",
        ))
        # Annotate peak events
        for event_name, color in [("Christmas Eve","#ef4444"),("Christmas Rush","#f59e0b"),
                                    ("Black Friday","#6366f1"),("Halloween","#22c55e")]:
            ev_rows = merged[merged["uk_event"] == event_name]
            if not ev_rows.empty:
                peak_row = ev_rows.loc[ev_rows["total_units"].idxmax()]
                fig.add_annotation(
                    x=peak_row["date"], y=peak_row["total_units"],
                    text=event_name, showarrow=True,
                    arrowhead=2, arrowcolor=color,
                    font=dict(color=color, size=11),
                    ax=-50, ay=-40,
                )
        fig.update_layout(
            title=dict(text="📈 Q4 2024 — Store Daily Sales Volume (SBY-LON-001)",
                       font=dict(color="#172b4d", size=14)),
            height=420, paper_bgcolor="#ffffff", plot_bgcolor="#f4f5f7",
            font=dict(color="#172b4d", family="Inter"),
            xaxis=dict(gridcolor="#dfe1e6"), yaxis=dict(gridcolor="#dfe1e6"),
            margin=dict(l=40, r=20, t=55, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Nectar promo impact
        nectar_impact = merged.groupby("is_nectar_week")["total_units"].mean().reset_index()
        nectar_impact["is_nectar_week"] = nectar_impact["is_nectar_week"].map(
            {0:"Non-Nectar Week", 1:"Nectar Promo Week"})
        nectar_impact["total_units"] = nectar_impact["total_units"].round(1)
        nectar_impact.columns = ["Week Type","Avg Daily Units"]

        st.markdown('<div class="section-header">🏷️ Nectar Price Promotion Impact</div>',
                    unsafe_allow_html=True)
        nc1, nc2 = st.columns(2)
        nc1.dataframe(nectar_impact, use_container_width=True, height=100)
        if len(nectar_impact) == 2:
            uplift = float(nectar_impact[nectar_impact["Week Type"]=="Nectar Promo Week"]["Avg Daily Units"].iloc[0])
            normal = float(nectar_impact[nectar_impact["Week Type"]=="Non-Nectar Week"]["Avg Daily Units"].iloc[0])
            pct = ((uplift - normal) / normal * 100) if normal > 0 else 0
            nc2.metric("Nectar Promo Uplift", f"+{pct:.1f}%",
                       f"+{uplift-normal:.0f} units/day on average")

    except Exception as e:
        st.error(f"Could not load calendar data: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — INTELLIGENCE HUB (Phase 4)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Intelligence Hub":
    st.markdown("# 🧠 Intelligence Hub")
    st.markdown(
        "<p style='color:#8b949e;'>Phase 4 — Weather impact, price elasticity, and RL decision agent.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    intel_tab = st.selectbox(
        "Intelligence Module",
        ["🌤️ Weather & Events", "📈 Price Elasticity", "🤖 RL Agent Status"]
    )

    # ── Weather & Events ────────────────────────────────────────────────────
    if intel_tab == "🌤️ Weather & Events":
        st.markdown('<div class="section-header">🌤️ Weather Impact Forecast</div>',
                    unsafe_allow_html=True)
        try:
            from external_factors import fetch_weather_forecast, get_weather_impact_summary, WEATHER_SENSITIVITY

            forecasts = fetch_weather_forecast()
            if forecasts:
                import plotly.graph_objects as go

                # Weather cards
                cols = st.columns(min(len(forecasts), 5))
                weather_icons = {
                    "Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️",
                    "Drizzle": "🌦️", "Snow": "❄️", "Fog": "🌫️",
                }
                for i, f in enumerate(forecasts[:5]):
                    icon = weather_icons.get(f["condition"], "🌡️")
                    cols[i].markdown(
                        f'<div class="kpi-card">'
                        f'<div style="font-size:2rem;">{icon}</div>'
                        f'<div class="kpi-value" style="font-size:1.4rem;">{f["temp_avg"]}°C</div>'
                        f'<div class="kpi-label">{f["date"]}<br>{f["condition"]} · {f["rain_mm"]}mm</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Impact analysis
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">📊 Predicted Category Demand Impact</div>',
                            unsafe_allow_html=True)

                summary = get_weather_impact_summary(forecasts)
                for day in summary:
                    if day["impacts"]:
                        with st.expander(f"📅 {day['date']} — {day['condition']} {day['temp_avg']}°C"):
                            for imp in day["impacts"]:
                                cats = ", ".join(imp["categories"])
                                st.markdown(
                                    f"**{imp['trigger']}** → Demand {imp['direction']} for: {cats}"
                                )
            else:
                st.info("Weather data unavailable. Set OPENWEATHER_API_KEY in .env or run with synthetic data.")

        except Exception as e:
            st.error(f"Weather module error: {e}")

        # Local Events
        st.markdown('<div class="section-header">📅 Upcoming Local Events</div>',
                    unsafe_allow_html=True)
        try:
            from external_factors import get_local_events
            from datetime import timedelta as td

            today = pd.Timestamp.today().date()
            event_rows = []
            for i in range(14):
                d = (today + td(days=i)).isoformat()
                events = get_local_events(d)
                for e in events:
                    event_rows.append({
                        "Date": d,
                        "Event": e["event_name"].replace("_", " ").title(),
                        "Footfall +": f"+{(e['footfall_multiplier']-1)*100:.0f}%",
                        "Affected Categories": ", ".join(e["affected_categories"]),
                    })
            if event_rows:
                st.dataframe(pd.DataFrame(event_rows), use_container_width=True, height=300)
            else:
                st.success("No major local events in the next 14 days.")
        except Exception as e:
            st.warning(f"Events module: {e}")

    # ── Price Elasticity ────────────────────────────────────────────────────
    elif intel_tab == "📈 Price Elasticity":
        st.markdown('<div class="section-header">📈 Price Elasticity by Category</div>',
                    unsafe_allow_html=True)
        try:
            from elasticity import PriceElasticityModel, get_category_elasticity_report

            model = PriceElasticityModel.load()
            if model.product_elasticities:
                report = get_category_elasticity_report(model)

                import plotly.graph_objects as go
                fig = go.Figure(go.Bar(
                    x=report["category"],
                    y=report["mean_elasticity"].abs(),
                    marker_color=[
                        "#ef4444" if abs(e) > 1.5 else "#f59e0b" if abs(e) > 1.0 else "#22c55e"
                        for e in report["mean_elasticity"]
                    ],
                    text=report["sensitivity"],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Price Sensitivity by Category (|Elasticity|)",
                    height=400, paper_bgcolor="#ffffff", plot_bgcolor="#f4f5f7",
                    font=dict(color="#172b4d", family="Inter"),
                    xaxis=dict(gridcolor="#dfe1e6"),
                    yaxis=dict(gridcolor="#dfe1e6", title="|Elasticity|"),
                    margin=dict(l=40, r=20, t=55, b=80),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown('<div class="section-header">🏷️ Nectar Price Simulation</div>',
                            unsafe_allow_html=True)
                st.dataframe(report, use_container_width=True, height=320)

                # Product-level lookup
                st.markdown('<div class="section-header">🔍 Product Elasticity Lookup</div>',
                            unsafe_allow_html=True)
                sel_pid = st.selectbox("Product ID", list(model.product_elasticities.keys())[:50])
                discount = st.slider("Nectar Discount %", 5, 40, 20) / 100
                if sel_pid:
                    uplift = model.predict_promo_uplift(sel_pid, discount)
                    data = model.product_elasticities[sel_pid]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Elasticity (ε)", f"{data['elasticity']:.2f}")
                    c2.metric("Predicted Uplift", f"+{uplift['predicted_uplift_pct']:.1f}%")
                    c3.metric("Confidence", uplift["confidence"].title())
            else:
                st.warning("Elasticity model not trained yet. Run: `python src/elasticity.py`")
        except Exception as e:
            st.error(f"Elasticity module error: {e}")

    # ── RL Agent Status ─────────────────────────────────────────────────────
    elif intel_tab == "🤖 RL Agent Status":
        st.markdown('<div class="section-header">🤖 Reinforcement Learning Decision Agent</div>',
                    unsafe_allow_html=True)
        try:
            from rl_agent import InventoryRLAgent

            agent = InventoryRLAgent.load()
            summary = agent.get_training_summary()

            if summary["status"] == "trained":
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Episodes Trained", f"{summary['episodes']:,}")
                c2.metric("Q-Table Size", f"{summary['q_table_size']:,}")
                c3.metric("Avg Reward (last 50)", f"{summary['final_avg_reward']:.1f}")
                c4.metric("Exploration Rate (ε)", f"{summary['epsilon']:.4f}")

                # Training curve
                if agent.episode_rewards:
                    import plotly.graph_objects as go
                    rewards = agent.episode_rewards
                    window = min(20, len(rewards))
                    smoothed = pd.Series(rewards).rolling(window, min_periods=1).mean()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=rewards, mode="lines", name="Episode Reward",
                        line=dict(color="rgba(240,106,0,0.2)", width=1),
                    ))
                    fig.add_trace(go.Scatter(
                        y=smoothed, mode="lines", name=f"Moving Avg ({window}ep)",
                        line=dict(color="#F06A00", width=2),
                    ))
                    fig.update_layout(
                        title="RL Agent Training Curve",
                        height=350, paper_bgcolor="#ffffff", plot_bgcolor="#f4f5f7",
                        font=dict(color="#172b4d", family="Inter"),
                        xaxis=dict(title="Episode", gridcolor="#dfe1e6"),
                        yaxis=dict(title="Reward", gridcolor="#dfe1e6"),
                        margin=dict(l=40, r=20, t=55, b=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Quick recommendation test
                st.markdown('<div class="section-header">🔍 Test RL Recommendation</div>',
                            unsafe_allow_html=True)
                tc1, tc2, tc3 = st.columns(3)
                test_stock = tc1.number_input("Stock on Hand", 0, 500, 10)
                test_risk = tc2.slider("Stockout Risk", 0.0, 1.0, 0.7)
                test_velocity = tc3.number_input("Velocity (7d)", 0.0, 100.0, 8.0)

                test_state = {
                    "stock_on_hand": test_stock,
                    "reorder_point": 30,
                    "sales_velocity_7d": test_velocity,
                    "stockout_probability": test_risk,
                    "velocity_trend": 0.1,
                    "weather_multiplier": 1.0,
                    "event_multiplier": 1.0,
                    "promo_demand_multiplier": 1.0,
                    "day_of_week": pd.Timestamp.today().dayofweek,
                }
                rec = agent.recommend_action(test_state)
                st.markdown(
                    f'<div class="insight-card">'
                    f'<div class="insight-product">🤖 Agent Decision: {rec["agent_recommendation"]}</div>'
                    f'<p>Order quantity: <b>{rec["order_qty"]} units</b> '
                    f'(multiplier: {rec["order_multiplier"]}x reorder point) · '
                    f'Confidence: {rec["confidence"]:.1%}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("RL agent not trained. Run: `python src/rl_agent.py`")
        except Exception as e:
            st.error(f"RL module error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — AUTO-ORDERS (Phase 4)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Auto-Orders":
    st.markdown("# 📦 Automated Purchase Orders")
    st.markdown(
        "<p style='color:#8b949e;'>Phase 4 — AI-generated replenishment orders with cost-benefit analysis.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    try:
        from auto_replenishment import (
            ReplenishmentEngine, generate_purchase_order_summary
        )
        prods_df = get_products()

        engine = ReplenishmentEngine(planning_horizon_days=7, service_level_target=0.95)
        orders = engine.generate_purchase_orders(df, prods_df)
        summary = generate_purchase_order_summary(orders)

        # Executive KPIs
        st.markdown('<div class="section-header">📊 Order Batch Summary</div>',
                    unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        for col, val, label in [
            (k1, summary["total_orders"], "Total Orders"),
            (k2, f'{summary["total_units"]:,}', "Total Units"),
            (k3, f'£{summary["total_value"]:,.0f}', "Order Value"),
            (k4, f'£{summary["revenue_protected"]:,.0f}', "Revenue Protected"),
            (k5, f'£{summary["net_benefit"]:,.0f}', "Net Benefit"),
        ]:
            col.markdown(
                f'<div class="kpi-card"><div class="kpi-value" style="font-size:1.5rem;">{val}</div>'
                f'<div class="kpi-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Priority breakdown
        st.markdown('<div class="section-header">🚨 Orders by Priority</div>',
                    unsafe_allow_html=True)
        priority_colors = {
            "ORDER NOW": "#ef4444",
            "Order Today": "#f59e0b",
            "Order This Week": "#22c55e",
            "Monitor": "#8b949e",
        }
        pc1, pc2, pc3, pc4 = st.columns(4)
        for col, priority in zip([pc1, pc2, pc3, pc4],
                                  ["ORDER NOW", "Order Today", "Order This Week", "Monitor"]):
            count = summary["priority_breakdown"].get(priority, 0)
            color = priority_colors.get(priority, "#8b949e")
            col.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value" style="font-size:1.8rem;-webkit-text-fill-color:{color};">{count}</div>'
                f'<div class="kpi-label">{priority}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Detailed orders table
        st.markdown('<div class="section-header">📋 Proposed Purchase Orders</div>',
                    unsafe_allow_html=True)

        f1, f2 = st.columns(2)
        rec_filter = f1.multiselect(
            "Priority Filter",
            ["ORDER NOW", "Order Today", "Order This Week", "Monitor"],
            default=["ORDER NOW", "Order Today", "Order This Week"],
        )
        cat_filter2 = f2.multiselect(
            "Category", sorted(orders["category"].unique()),
            default=list(orders["category"].unique()),
        )

        filtered_orders = orders[
            orders["recommendation"].isin(rec_filter) &
            orders["category"].isin(cat_filter2)
        ].copy()

        display_orders = filtered_orders[[
            "recommendation", "product_name", "category",
            "order_qty", "order_value", "urgency_score",
            "stock_on_hand", "days_of_cover_current", "days_of_cover_after_order",
            "holding_cost", "lost_revenue_if_no_order", "net_benefit",
        ]].copy()
        display_orders.columns = [
            "Priority", "Product", "Category",
            "Order Qty", "Order £", "Urgency",
            "Current Stock", "Days Cover (Now)", "Days Cover (After)",
            "Holding Cost £", "Revenue at Risk £", "Net Benefit £",
        ]
        display_orders["Order £"] = display_orders["Order £"].round(2)
        display_orders["Urgency"] = display_orders["Urgency"].round(1)
        display_orders["Net Benefit £"] = display_orders["Net Benefit £"].round(2)

        st.dataframe(display_orders.reset_index(drop=True),
                     use_container_width=True, height=480)

        st.markdown(
            f"<p style='color:#484f58;font-size:0.78rem;'>"
            f"Showing {len(filtered_orders)} of {len(orders)} orders · "
            f"Generated at {summary['generated_at'][:19]}</p>",
            unsafe_allow_html=True,
        )

        # Cost-benefit chart
        if not filtered_orders.empty:
            import plotly.graph_objects as go
            top20 = filtered_orders.head(20)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top20["product_name"], y=top20["lost_revenue_if_no_order"],
                name="Revenue at Risk £", marker_color="#ef4444",
            ))
            fig.add_trace(go.Bar(
                x=top20["product_name"], y=top20["holding_cost"],
                name="Holding Cost £", marker_color="#f59e0b",
            ))
            fig.add_trace(go.Bar(
                x=top20["product_name"], y=top20["net_benefit"],
                name="Net Benefit £", marker_color="#22c55e",
            ))
            fig.update_layout(
                title="Cost-Benefit Analysis — Top 20 Orders",
                barmode="group", height=420,
                paper_bgcolor="#ffffff", plot_bgcolor="#f4f5f7",
                font=dict(color="#172b4d", family="Inter"),
                xaxis=dict(gridcolor="#dfe1e6", tickangle=-45),
                yaxis=dict(gridcolor="#dfe1e6", title="£ GBP"),
                margin=dict(l=40, r=20, t=55, b=120),
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Auto-replenishment module error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — DEMAND FORECAST (Phase 5)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Demand Forecast":
    st.markdown("# 📈 Demand Forecasting")
    st.markdown(
        "<p style='color:#8b949e;'>Phase 5 — Holt-Winters + seasonal demand forecasting with event uplift.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    try:
        from forecaster import DemandForecaster

        @st.cache_resource(show_spinner=False)
        def load_forecaster():
            return DemandForecaster.load_or_train()

        with st.spinner("Loading demand forecasting engine …"):
            forecaster = load_forecaster()

        prods_df = get_products()
        cats = sorted(prods_df["category"].unique())

        forecast_tab = st.selectbox(
            "View",
            ["📊 Category Overview", "🔍 Product Deep-Dive", "📋 All Products Table"]
        )

        # ── Category Overview ─────────────────────────────────────────────
        if forecast_tab == "📊 Category Overview":
            st.markdown('<div class="section-header">📊 30-Day Category Demand Forecast</div>',
                        unsafe_allow_html=True)
            horizon = st.slider("Forecast Horizon (days)", 7, 90, 30, step=7)

            cat_rows = []
            for cat in cats:
                cfc = forecaster.forecast_category(cat, horizon=horizon)
                if "error" not in cfc:
                    s = cfc["summary"]
                    cat_rows.append({
                        "Category": cat,
                        "Products": cfc["n_products"],
                        "Total Demand": int(s["total_demand"]),
                        "Avg Daily": round(s["avg_daily"], 1),
                        "Peak Day": s["peak_day"],
                    })

            if cat_rows:
                cat_df = pd.DataFrame(cat_rows).sort_values("Total Demand", ascending=False)

                import plotly.graph_objects as go
                fig_cat = go.Figure(go.Bar(
                    x=cat_df["Category"],
                    y=cat_df["Total Demand"],
                    marker_color=[
                        "#F06A00" if i < 3 else "#7B2D8B" if i < 6 else "#3d5a80"
                        for i in range(len(cat_df))
                    ],
                    text=cat_df["Total Demand"].apply(lambda x: f"{x:,}"),
                    textposition="outside",
                ))
                fig_cat.update_layout(
                    title=f"{horizon}-Day Demand Forecast by Category",
                    height=420, paper_bgcolor="#ffffff", plot_bgcolor="#f4f5f7",
                    font=dict(color="#172b4d", family="Inter"),
                    xaxis=dict(gridcolor="#dfe1e6", tickangle=-25),
                    yaxis=dict(gridcolor="#dfe1e6", title="Forecasted Units"),
                    margin=dict(l=40, r=20, t=55, b=100),
                )
                st.plotly_chart(fig_cat, use_container_width=True)
                st.dataframe(cat_df, use_container_width=True, height=280)

        # ── Product Deep-Dive ─────────────────────────────────────────────
        elif forecast_tab == "🔍 Product Deep-Dive":
            st.markdown('<div class="section-header">🔍 Single Product Forecast</div>',
                        unsafe_allow_html=True)

            cat_sel = st.selectbox("Filter by Category", ["All"] + cats)
            pids = list(forecaster.product_models.keys())
            if cat_sel != "All":
                pids = [p for p in pids if forecaster.category_map.get(p) == cat_sel]

            pid_sel = st.selectbox("Product ID", pids[:100])
            horizon2 = st.slider("Horizon", 7, 90, 30, step=7, key="h2")

            if pid_sel:
                fc_result = forecaster.forecast_product(pid_sel, horizon=horizon2)
                if "error" not in fc_result:
                    s = fc_result["summary"]
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("30-Day Total", f"{s['total_demand_30d']:,.0f} units")
                    m2.metric("Avg Daily", f"{s['avg_daily']} units")
                    m3.metric("Peak Day", s['peak_day'].split('-')[2] + "/" + s['peak_day'].split('-')[1])
                    m4.metric("Historical Avg", f"{s['base_demand_historical']} units")

                    daily = pd.DataFrame(fc_result["daily_forecasts"])
                    daily["date"] = pd.to_datetime(daily["date"])

                    import plotly.graph_objects as go
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(
                        x=daily["date"], y=daily["upper_90"],
                        mode="lines", name="Upper 90% CI",
                        line=dict(width=0), showlegend=False,
                        fillcolor="rgba(240,106,0,0.15)",
                    ))
                    fig_p.add_trace(go.Scatter(
                        x=daily["date"], y=daily["lower_90"],
                        mode="lines", name="Lower 90% CI",
                        line=dict(width=0),
                        fill="tonexty", fillcolor="rgba(240,106,0,0.15)",
                        showlegend=False,
                    ))
                    fig_p.add_trace(go.Scatter(
                        x=daily["date"], y=daily["forecast"],
                        mode="lines+markers", name="Forecast",
                        line=dict(color="#F06A00", width=2.5),
                        marker=dict(size=4),
                    ))
                    # Annotate events
                    events = daily[daily["event"].notna()]
                    if not events.empty:
                        fig_p.add_trace(go.Scatter(
                            x=events["date"], y=events["forecast"],
                            mode="markers", name="Event Day",
                            marker=dict(color="#7B2D8B", size=10, symbol="star"),
                        ))
                    fig_p.update_layout(
                        title=f"Demand Forecast — {pid_sel} ({fc_result['category']})",
                        height=420, paper_bgcolor="#ffffff", plot_bgcolor="#f4f5f7",
                        font=dict(color="#172b4d", family="Inter"),
                        xaxis=dict(gridcolor="#dfe1e6"),
                        yaxis=dict(gridcolor="#dfe1e6", title="Units"),
                        margin=dict(l=40, r=20, t=55, b=40),
                        legend=dict(orientation="h", y=1.12),
                    )
                    st.plotly_chart(fig_p, use_container_width=True)

                    # Event table
                    event_days = daily[daily["event"].notna()][["date","event","event_multiplier","forecast"]]
                    if not event_days.empty:
                        st.markdown('<div class="section-header">🗓️ Event-Adjusted Days</div>',
                                    unsafe_allow_html=True)
                        event_days["date"] = event_days["date"].dt.strftime("%Y-%m-%d")
                        event_days.columns = ["Date","Event","Multiplier","Forecast (units)"]
                        st.dataframe(event_days.reset_index(drop=True),
                                     use_container_width=True, height=200)

        # ── All Products Table ────────────────────────────────────────────
        elif forecast_tab == "📋 All Products Table":
            st.markdown('<div class="section-header">📋 All Products — 30-Day Forecast Summary</div>',
                        unsafe_allow_html=True)

            cat_f = st.selectbox("Category Filter", ["All"] + cats, key="fc_cat")
            with st.spinner("Building forecast summaries …"):
                summaries = forecaster.get_all_product_summaries(horizon=30)

            if cat_f != "All":
                summaries = [s for s in summaries if s["category"] == cat_f]

            if summaries:
                fc_df = pd.DataFrame(summaries)
                fc_df["vs_historical"] = fc_df["vs_historical"].apply(
                    lambda x: f"{x:+.1f}%"
                )
                fc_df.columns = [
                    "Product ID", "Category", "30d Total", "Avg Daily",
                    "Peak Day", "Peak Units", "vs Historical", "Training Days"
                ]
                st.dataframe(fc_df.reset_index(drop=True), use_container_width=True, height=520)
                st.markdown(
                    f"<p style='color:#484f58;font-size:0.78rem;'>Showing {len(summaries)} products</p>",
                    unsafe_allow_html=True,
                )

    except FileNotFoundError:
        st.warning("Forecasting model not found. Run: `python src/forecaster.py`")
    except Exception as e:
        st.error(f"Forecasting error: {e}")
        import traceback; st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — CO-PILOT (Phase 5)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Co-Pilot":
    st.markdown("# 🤖 RetailBrain Co-Pilot")
    st.markdown(
        "<p style='color:#8b949e;'>Phase 5 — Conversational AI assistant for store managers with live retail context.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    try:
        from copilot import RetailCopilot
        from anomaly_detector import AnomalyDetector
        from data_ingestion import load_sales, load_inventory, load_calendar

        @st.cache_resource(show_spinner=False)
        def load_copilot_resources():
            """Load anomaly detector and co-pilot (cached at session level)."""
            from forecaster import DemandForecaster
            det = AnomalyDetector()
            with st.spinner("Analysing demand patterns …"):
                det.fit(load_sales(), load_inventory(), load_calendar(), get_products())
            fc = DemandForecaster.load_or_train()
            return det, fc

        with st.spinner("Initialising Co-Pilot …"):
            det, fc = load_copilot_resources()
            copilot = RetailCopilot()

            # Inject live context
            risk_df = df.copy()
            risk_context = []
            for _, row in risk_df.iterrows():
                risk_context.append({
                    "product_id": row.get("product_id", ""),
                    "product_name": row.get("product_name", ""),
                    "category": row.get("category", ""),
                    "stockout_risk": float(row.get("stockout_probability", 0)),
                    "units_on_hand": float(row.get("stock_on_hand", 0)),
                    "units_on_order": float(row.get("units_on_order", 0) if "units_on_order" in row else 0),
                    "reorder_point": float(row.get("reorder_point", 0)),
                    "days_of_cover": float(row.get("days_of_cover", 0)),
                })
            anomaly_list = det.get_recent_anomalies(days=30)
            fc_summaries = fc.get_all_product_summaries(horizon=30)
            copilot.set_context_data(
                risk_data=risk_context,
                anomaly_data=anomaly_list,
                forecast_summary=fc_summaries,
            )

        # Model status banner
        model_label = "🟢 GPT-4o-mini" if copilot._has_openai else "🟡 Rule-Based Engine"
        st.markdown(
            f"<div style='background:#ffffff;border:1px solid #dfe1e6;border-left:4px solid "
            f"{'#22c55e' if copilot._has_openai else '#f59e0b'};"
            f"border-radius:8px;padding:10px 16px;margin-bottom:16px;color:#172b4d;'>"
            f"<b>Model:</b> {model_label} &nbsp;·&nbsp; "
            f"<b>Context:</b> {len(risk_context)} products · {len(anomaly_list)} anomalies · {len(fc_summaries)} forecasts"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Conversation history
        if "copilot_history" not in st.session_state:
            st.session_state.copilot_history = []

        # Quick-starter buttons
        st.markdown('<div class="section-header">💬 Quick Questions</div>',
                    unsafe_allow_html=True)
        q_cols = st.columns(4)
        quick_questions = [
            "What should I order this week?",
            "Show me the top 5 stockout risks",
            "Are there any demand anomalies?",
            "What's the 30-day demand forecast?",
        ]
        for col, qq in zip(q_cols, quick_questions):
            if col.button(qq, use_container_width=True):
                st.session_state.copilot_pending_question = qq

        st.markdown("---")

        # Chat conversation display
        for msg in st.session_state.copilot_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                st.markdown(
                    f"<div style='background:#fff7ed;border-radius:8px;padding:12px 16px;"
                    f"margin-bottom:10px;border-left:3px solid #F06A00;border:1px solid #fed7aa;'>"
                    f"<b style='color:#F06A00;'>You</b><br><span style='color:#172b4d;'>{content}</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='background:#ffffff;border:1px solid #dfe1e6;border-radius:8px;"
                    f"padding:14px 18px;margin-bottom:14px;'>"
                    f"<b style='color:#7B2D8B;'>🤖 Co-Pilot</b><br><br><span style='color:#172b4d;'>{content}</span></div>",
                    unsafe_allow_html=True,
                )

        # Input area
        with st.form("copilot_form", clear_on_submit=True):
            user_q = st.text_input(
                "Ask Co-Pilot anything about your retail operations …",
                value=st.session_state.pop("copilot_pending_question", ""),
                placeholder="e.g. What should I order this week? / Why is demand spiking on SAI-ABCD?",
            )
            col_send, col_clear = st.columns([5, 1])
            submitted = col_send.form_submit_button("Send 📨", use_container_width=True)
            cleared = col_clear.form_submit_button("Clear 🗑️", use_container_width=True)

        if cleared:
            st.session_state.copilot_history = []
            copilot.clear_history()
            st.rerun()

        if submitted and user_q.strip():
            with st.spinner("Co-Pilot is thinking …"):
                result = copilot.ask(user_q.strip())

            st.session_state.copilot_history.append({"role": "user", "content": user_q.strip()})
            st.session_state.copilot_history.append({"role": "assistant", "content": result["answer"]})

            # Show meta info
            st.markdown(
                f"<p style='color:#484f58;font-size:0.72rem;'>"
                f"Intent: {result['intent']} · Model: {result['model_used']} · "
                f"Sources: {', '.join(result['sources_used'])}</p>",
                unsafe_allow_html=True,
            )
            st.rerun()

        # Anomaly sidebar
        if anomaly_list:
            st.markdown("---")
            st.markdown('<div class="section-header">🔍 Recent Anomalies (Last 30 Days)</div>',
                        unsafe_allow_html=True)
            high_anom = [a for a in anomaly_list if a["severity"] == "high"][:8]
            if high_anom:
                for a in high_anom:
                    sev_color = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}.get(a["severity"], "#8b949e")
                    st.markdown(
                        f"<div style='background:#ffffff;border:1px solid #dfe1e6;"
                        f"border-left:3px solid {sev_color};border-radius:8px;"
                        f"padding:10px 14px;margin-bottom:8px;font-size:0.85rem;color:#172b4d;'>"
                        f"<b>{a['date']}</b> · <code>{a['product_id']}</code> · "
                        f"<span style='color:{sev_color};'>{a['anomaly_type'].replace('_',' ').title()}</span>"
                        f"<br><span style='color:#5e6c84;'>{a['description'][:100]}…</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.success("No high-severity anomalies in the last 30 days.")

    except Exception as e:
        st.error(f"Co-Pilot error: {e}")
        import traceback; st.code(traceback.format_exc())
