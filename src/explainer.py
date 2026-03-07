"""
Retail Brain — LLM Explanation Layer
Generates human-readable, manager-friendly explanations for stockout predictions.
Falls back to rule-based templates if OPENAI_API_KEY is not set.
"""

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ── Rule-based fallback templates ─────────────────────────────────────────────
_TEMPLATES = {
    "High": (
        "⚠️ {product} is at high risk of stocking out in approximately {time_to_stockout}. "
        "Current stock ({stock:.0f} units) is critically low relative to the recent 7-day "
        "sales velocity of {velocity:.1f} units/day. "
        "{promo_note}"
        "Immediate replenishment of {qty} units is strongly recommended to avoid lost sales."
    ),
    "Medium": (
        "📋 {product} shows moderate stockout risk over the next 72 hours. "
        "With {stock:.0f} units remaining and a 7-day sales velocity of {velocity:.1f} units/day, "
        "stock should last approximately {time_to_stockout}. "
        "{promo_note}"
        "Ordering {qty} units within the next day is advisable."
    ),
    "Low": (
        "✅ {product} has adequate inventory coverage. "
        "Current stock of {stock:.0f} units against a sales velocity of {velocity:.1f} units/day "
        "provides approximately {time_to_stockout} of coverage. No immediate action required."
    ),
}

_PROMO_NOTE = "An active or recent promotional period is contributing to higher-than-usual demand. "
_TREND_NOTE = "Sales velocity is trending {direction} compared to the 14-day average. "


def _rule_based_explanation(row: dict) -> str:
    """Generate a template-based explanation from product row data."""
    risk    = row.get("risk_level", "Low")
    promo   = row.get("promo_days_last_7", 0) > 0
    trend   = row.get("velocity_trend", 0.0)

    promo_note = _PROMO_NOTE if promo else ""
    trend_note = ""
    if abs(trend) > 0.1:
        direction  = "upward" if trend > 0 else "downward"
        trend_note = _TREND_NOTE.format(direction=direction)

    template = _TEMPLATES[risk]
    return template.format(
        product         = row.get("product_name", "This product"),
        stock           = row.get("stock_on_hand", 0),
        velocity        = row.get("sales_velocity_7d", 0),
        time_to_stockout= row.get("time_to_stockout", "N/A"),
        qty             = row.get("replenishment_qty", 0),
        promo_note      = promo_note + trend_note,
    )


def _llm_explanation(row: dict) -> str:
    """Call OpenAI to generate a 2-3 sentence plain-English explanation."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        system_prompt = (
            "You are an AI retail operations assistant. Your role is to explain "
            "stockout predictions to store managers in clear, concise, professional English. "
            "Write 2-3 sentences only. Be specific about numbers and time frames. "
            "Do not use bullet points or headers."
        )

        user_prompt = f"""
Product: {row.get('product_name', 'Unknown')}
Stockout Risk: {row.get('risk_level', 'Unknown')} ({row.get('stockout_probability', 0)*100:.0f}%)
Current Stock: {row.get('stock_on_hand', 0):.0f} units
Days of Cover: {row.get('days_of_cover', 0):.1f} days
7-Day Sales Velocity: {row.get('sales_velocity_7d', 0):.1f} units/day
Velocity Trend vs 14d: {row.get('velocity_trend', 0)*100:+.1f}%
Promotion Active/Recent: {'Yes' if row.get('promo_days_last_7', 0) > 0 else 'No'}
Recommended Action: {row.get('recommended_action', 'N/A')}
Replenishment Quantity: {row.get('replenishment_qty', 0)} units

Write a brief explanation for the store manager about why this product has been flagged.
"""
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        # Graceful fallback
        return _rule_based_explanation(row) + f"\n_(LLM unavailable: {e})_"


def generate_explanation(row: dict) -> str:
    """
    Generate a human-readable explanation for a single product's stockout risk.
    Uses OpenAI if API key is available, otherwise falls back to rule-based templates.
    """
    if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
        return _llm_explanation(row)
    return _rule_based_explanation(row)


def generate_explanations_batch(df, max_products: int = 10) -> dict[str, str]:
    """
    Generate explanations for multiple products.
    Returns a dict: {product_id: explanation_string}
    """
    results = {}
    for _, row in df.head(max_products).iterrows():
        pid = row.get("product_id", row.get("product_name", "unknown"))
        results[pid] = generate_explanation(row.to_dict())
    return results


if __name__ == "__main__":
    sample_row = {
        "product_name": "Whole Milk 2L",
        "risk_level": "High",
        "stockout_probability": 0.87,
        "stock_on_hand": 6,
        "days_of_cover": 0.7,
        "sales_velocity_7d": 9.2,
        "velocity_trend": 0.18,
        "promo_days_last_7": 2,
        "recommended_action": "Replenish immediately",
        "replenishment_qty": 60,
        "time_to_stockout": "17 hours",
    }
    print("Rule-based explanation:")
    print(_rule_based_explanation(sample_row))

    if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
        print("\nLLM explanation:")
        print(_llm_explanation(sample_row))
    else:
        print("\n(Set OPENAI_API_KEY in .env to enable LLM explanations)")
