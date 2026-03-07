"""
Retail Brain — Plotly Chart Helpers
Reusable chart functions for the Streamlit dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


PALETTE = {
    "High":   "#ef4444",
    "Medium": "#f59e0b",
    "Low":    "#22c55e",
    "bg":     "#0f172a",
    "card":   "#1e293b",
    "accent": "#6366f1",
    "text":   "#f8fafc",
    "muted":  "#94a3b8",
}


def _base_layout(title: str = "", height: int = 380) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=PALETTE["text"], size=14)),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor=PALETTE["card"],
        plot_bgcolor=PALETTE["card"],
        font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
    )


def risk_bar_chart(df: pd.DataFrame, n: int = 12) -> go.Figure:
    """Horizontal bar chart of top-N products by stockout probability."""
    top = df.sort_values("stockout_probability", ascending=False).head(n).copy()
    top["label"] = top.apply(
        lambda r: f"{r.get('product_name','?')} ({r['stockout_probability']*100:.0f}%)", axis=1
    )
    colors = top["risk_level"].map(PALETTE).tolist()

    fig = go.Figure(go.Bar(
        x=top["stockout_probability"] * 100,
        y=top["label"],
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Risk: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout("🔝 Top Products by Stockout Risk", height=420),
        xaxis_title="Risk Probability (%)",
    )
    fig.update_yaxes(autorange="reversed", gridcolor="#334155")
    fig.add_vline(x=70, line_dash="dot", line_color=PALETTE["High"],
                  annotation_text="High", annotation_font_color=PALETTE["High"])
    fig.add_vline(x=40, line_dash="dot", line_color=PALETTE["Medium"],
                  annotation_text="Medium", annotation_font_color=PALETTE["Medium"])
    return fig


def risk_distribution_pie(df: pd.DataFrame) -> go.Figure:
    """Donut chart of risk level distribution across all products."""
    counts = df["risk_level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.65,
        marker_colors=[PALETTE["High"], PALETTE["Medium"], PALETTE["Low"]],
        hovertemplate="<b>%{label}</b>: %{value} products (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout("📊 Risk Distribution"),
        showlegend=True,
        legend=dict(font=dict(color=PALETTE["text"])),
    )
    fig.add_annotation(
        text=f"<b>{len(df)}</b><br>Products",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color=PALETTE["text"]),
    )
    return fig


def sales_trend_chart(sales_df: pd.DataFrame, product_id: str, product_name: str) -> go.Figure:
    """90-day daily sales line chart for a single product."""
    prod_sales = sales_df[sales_df["product_id"] == product_id].sort_values("date").copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prod_sales["date"],
        y=prod_sales["units_sold"],
        mode="lines",
        name="Daily Sales",
        line=dict(color=PALETTE["accent"], width=2),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.1)",
    ))
    # Rolling 7-day average
    prod_sales["ma7"] = prod_sales["units_sold"].rolling(7, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=prod_sales["date"],
        y=prod_sales["ma7"],
        mode="lines",
        name="7-Day Avg",
        line=dict(color="#f59e0b", width=2, dash="dash"),
    ))
    fig.update_layout(
        **_base_layout(f"📈 Sales Trend — {product_name}"),
        xaxis_title="Date",
        yaxis_title="Units Sold",
        legend=dict(font=dict(color=PALETTE["text"])),
    )
    return fig


def stock_trend_chart(inventory_df: pd.DataFrame, product_id: str, product_name: str,
                       reorder_point: float = 20) -> go.Figure:
    """90-day stock-on-hand trend with reorder point reference line."""
    prod_inv = inventory_df[inventory_df["product_id"] == product_id].sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prod_inv["date"],
        y=prod_inv["stock_on_hand"],
        mode="lines",
        name="Stock on Hand",
        line=dict(color="#22c55e", width=2),
        fill="tozeroy",
        fillcolor="rgba(34,197,94,0.1)",
    ))
    fig.add_hline(
        y=reorder_point,
        line_dash="dash", line_color=PALETTE["Medium"],
        annotation_text=f"Reorder Point ({reorder_point:.0f})",
        annotation_font_color=PALETTE["Medium"],
    )
    fig.update_layout(
        **_base_layout(f"📦 Stock Trend — {product_name}"),
        xaxis_title="Date",
        yaxis_title="Units on Hand",
    )
    return fig


def risk_gauge(probability: float, product_name: str) -> go.Figure:
    """Gauge chart showing a product's stockout risk probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title=dict(text=f"Stockout Risk<br>{product_name}", font=dict(color=PALETTE["text"])),
        number=dict(suffix="%", font=dict(color=PALETTE["text"])),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=PALETTE["text"]),
            bar=dict(color=PALETTE["High"] if probability >= 0.7
                          else PALETTE["Medium"] if probability >= 0.4
                          else PALETTE["Low"]),
            steps=[
                dict(range=[0,   40], color="#1e3a2f"),
                dict(range=[40,  70], color="#3b2a0e"),
                dict(range=[70, 100], color="#3b0e0e"),
            ],
            threshold=dict(
                line=dict(color="white", width=2),
                thickness=0.75,
                value=probability * 100,
            ),
            bgcolor=PALETTE["card"],
        ),
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=30, r=30, t=60, b=20),
        paper_bgcolor=PALETTE["card"],
        font=dict(color=PALETTE["text"]),
    )
    return fig
