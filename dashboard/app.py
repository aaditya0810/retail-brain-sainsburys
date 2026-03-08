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

# ── CSS — Sainsbury's brand palette (orange #F06A00, purple #7B2D8B) ───────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Background */
  .stApp { background: #0d1117; color: #f0f6fc; }

  /* Sidebar — Sainsbury's deep purple/dark */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1033 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
  }
  [data-testid="stSidebar"] * { color: #f0f6fc !important; }

  /* Brand bar at top of sidebar */
  .brand-bar {
    background: linear-gradient(135deg, #F06A00 0%, #E8521E 50%, #7B2D8B 100%);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
  }
  .brand-title { font-size: 1.1rem; font-weight: 800; color: white; }
  .brand-sub   { font-size: 0.72rem; color: rgba(255,255,255,0.8); margin-top:2px; }

  /* KPI Cards */
  .kpi-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 20px 22px;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
  }
  .kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 28px rgba(240,106,0,.20);
  }
  .kpi-value {
    font-size: 2.0rem; font-weight: 700;
    background: linear-gradient(135deg, #F06A00, #f5a623);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .kpi-label { font-size: 0.8rem; color: #8b949e; margin-top: 4px; }

  /* Tier badges */
  .badge-ttd  { background:#7B2D8B22; color:#c77dff; border:1px solid #7B2D8B55;
                padding:2px 9px; border-radius:99px; font-size:.75rem; font-weight:600; }
  .badge-std  { background:#F06A0022; color:#F06A00; border:1px solid #F06A0055;
                padding:2px 9px; border-radius:99px; font-size:.75rem; font-weight:600; }
  .badge-branded { background:#0d419d22; color:#58a6ff; border:1px solid #0d419d55;
                padding:2px 9px; border-radius:99px; font-size:.75rem; font-weight:600; }

  /* Risk badges */
  .badge-danger  { background:#ef444422; color:#ef4444; border:1px solid #ef444455;
                   padding:2px 9px; border-radius:99px; font-size:.78rem; font-weight:600; }
  .badge-warning { background:#f59e0b22; color:#f59e0b; border:1px solid #f59e0b55;
                   padding:2px 9px; border-radius:99px; font-size:.78rem; font-weight:600; }
  .badge-success { background:#22c55e22; color:#22c55e; border:1px solid #22c55e55;
                   padding:2px 9px; border-radius:99px; font-size:.78rem; font-weight:600; }

  /* Insight cards */
  .insight-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #F06A00;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
    line-height: 1.65;
    color: #e6edf3;
  }
  .insight-product { font-size: 1.0rem; font-weight:600; color:#f5a623; margin-bottom:6px; }

  /* Event pill */
  .event-pill {
    background: rgba(240,106,0,0.15);
    border: 1px solid rgba(240,106,0,0.4);
    color: #F06A00;
    padding: 3px 12px; border-radius: 99px; font-size:0.78rem; font-weight:600;
  }

  /* Section headers */
  .section-header {
    font-size: 1.15rem; font-weight:600; color:#f0f6fc;
    margin: 18px 0 10px 0;
    border-bottom: 1px solid #21262d; padding-bottom:7px;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #F06A00, #E8521E);
    color:white; border:none; border-radius:8px; font-weight:500;
    transition: opacity .2s;
  }
  .stButton > button:hover { opacity:.85; }
  .stApp { scrollbar-color: #F06A00 #161b22; }
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
    st.markdown("""
    <div class="brand-bar">
      <div class="brand-title">🛒 Retail Brain</div>
      <div class="brand-sub">Sainsbury's × AI Stockout Intelligence</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<p style='color:#8b949e;font-size:0.75rem;'>Store: <b style='color:#F06A00'>SBY-LON-001</b> · Q4 2024</p>",
                unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📋 Risk Table", "🔍 Product Detail",
         "💡 Manager Insights", "📅 Event Calendar"],
        label_visibility="collapsed",
    )
    st.divider()
    if st.button("🔄 Refresh Predictions"):
        st.cache_data.clear()
        st.rerun()
    st.markdown(
        "<p style='color:#21262d;font-size:0.7rem;margin-top:8px;'>"
        "© 2024 Sainsbury's PLC · Retail Brain v1.0<br>"
        "Powered by XGBoost + OpenAI</p>",
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
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(240,106,0,0.1) 0%, rgba(123,45,139,0.1) 100%);
                    border-radius: 16px; padding: 32px 36px; border: 1px solid rgba(240,106,0,0.2); 
                    margin-bottom: 24px; position: relative; overflow: hidden;
                    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1); backdrop-filter: blur(5px);">
            <div style="position: absolute; right: -20px; top: -40px; opacity: 0.1; font-size: 14rem;">🛒</div>
            <h1 style="color: #F06A00; margin:0 0 12px 0; font-size: 2.6rem; font-weight: 800; letter-spacing: -0.5px;">
                Retail Brain <span style="color:#e6edf3; font-weight:300;">| Stockout Intelligence</span>
            </h1>
            <p style="color: #8b949e; font-size: 1.15rem; margin: 0 0 16px 0; max-width: 700px; line-height: 1.5;">
                Advanced predictive analytics engine monitoring <strong>Sainsbury's SBY-LON-001 (London Flagship)</strong>. 
                Utilising XGBoost to forecast inventory depletion risks across 76 premium SKUs during Q4 peak trading.
            </p>
            <div style="display: flex; gap: 12px;">
               <span style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; color:#8b949e;">Q4 2024 Simulation</span>
               <span style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; color:#8b949e;">Live Decision Intelligence</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Active UK event banner
    try:
        sales = get_raw_sales()
        latest_date = pd.to_datetime(sales["date"]).max()
        latest_event = sales[sales["date"] == latest_date]["uk_event"].mode()[0]
        if latest_event != "Normal":
            st.markdown(
                f'<div style="margin:8px 0 16px 0;">'
                f'<span class="event-pill">🗓️ Active Trading Event: {latest_event}</span>'
                f'</div>', unsafe_allow_html=True
            )
    except Exception:
        pass

    st.markdown("---")

    # KPIs
    total     = len(df)
    n_high    = int((df["risk_level"] == "High").sum())
    n_medium  = int((df["risk_level"] == "Medium").sum())
    avg_doc   = df["days_of_cover"].clip(upper=999).mean()
    avg_risk  = df["stockout_probability"].mean() * 100
    prods_df  = get_products()
    n_ttd     = int((prods_df["tier"] == "Taste the Difference").sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, val, label, color in [
        (c1, total,    "Total SKUs",          None),
        (c2, n_high,   "🔴 High Risk",         "#ef4444"),
        (c3, n_medium, "🟡 Medium Risk",        "#f59e0b"),
        (c4, f"{avg_doc:.1f}", "Avg Days Cover", None),
        (c5, f"{avg_risk:.0f}%", "Avg Risk Score", None),
        (c6, n_ttd,    "Taste the Diff SKUs",  "#c77dff"),
    ]:
        style = f'style="-webkit-text-fill-color:{color};"' if color else ""
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-value" {style}>{val}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_bar, col_pie = st.columns([2, 1])
    with col_bar:
        st.plotly_chart(risk_bar_chart(df, n=15), use_container_width=True)
    with col_pie:
        st.plotly_chart(risk_distribution_pie(df), use_container_width=True)

    # Tier breakdown
    st.markdown('<div class="section-header">📊 Risk Breakdown by Product Tier</div>',
                unsafe_allow_html=True)
    merged = df.merge(prods_df[["product_id","tier"]], on="product_id", how="left")
    tier_risk = merged.groupby("tier")["stockout_probability"].agg(["mean","count"]).reset_index()
    tier_risk.columns = ["Tier", "Avg Risk %", "Products"]
    tier_risk["Avg Risk %"] = (tier_risk["Avg Risk %"] * 100).round(1)
    tier_risk["Tier Badge"] = tier_risk["Tier"].map({
        "Taste the Difference": "🟣 Taste the Difference",
        "Sainsbury's":          "🟠 Sainsbury's",
        "Branded":              "🔵 Branded",
    })
    st.dataframe(tier_risk[["Tier Badge","Products","Avg Risk %"]].sort_values("Avg Risk %", ascending=False),
                 use_container_width=True, height=160)

    # Alert banner
    high_risk = df[df["risk_level"] == "High"].sort_values("stockout_probability", ascending=False)
    if not high_risk.empty:
        st.markdown('<div class="section-header">🚨 Immediate Replenishment Required</div>',
                    unsafe_allow_html=True)
        for _, row in high_risk.head(6).iterrows():
            ca, cb, cc, cd, ce = st.columns([3.5, 1.2, 1.2, 1.5, 2])
            ca.markdown(f"**{row['product_name']}**")
            cb.markdown(f"🔴 `{row['stockout_probability']*100:.0f}%`")
            cc.markdown(f"⏱️ {row['time_to_stockout']}")
            cd.markdown(f"📦 Order **{row['replenishment_qty']}** units")
            price_row = prods_df[prods_df["product_id"] == row.get("product_id","")]
            if not price_row.empty:
                rev_risk = round(row["replenishment_qty"] * float(price_row.iloc[0]["unit_price"]), 2)
                ce.markdown(f"💷 Revenue at risk: **£{rev_risk:,.2f}**")
            st.markdown("<hr style='margin:3px 0;border-color:#21262d;'>", unsafe_allow_html=True)


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
                       font=dict(color="#f0f6fc", size=14)),
            height=420, paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            font=dict(color="#f0f6fc", family="Inter"),
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
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
