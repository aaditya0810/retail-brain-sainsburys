"""
Retail Brain — LLM Co-Pilot (Phase 5)
Conversational AI assistant for store managers and supply chain analysts.

Features:
  - RAG (Retrieval Augmented Generation): injects live retail context into prompts
  - OpenAI GPT-4o-mini when OPENAI_API_KEY is set
  - Rule-based deterministic fallback when no API key provided
  - Question categorisation and intent detection
  - Markdown-formatted responses with actionable recommendations

Supported question types:
  - Risk explanation: "why is X at risk?"
  - Order recommendation: "what should I order this week?"
  - Anomaly insight: "what's driving the demand spike?"
  - Category summary: "how is frozen food performing?"
  - Forecast query: "what's expected demand for Christmas?"
  - General ops: "show me the top 5 stockout risks"
"""

import os
import sys
import json
import re
import warnings
from datetime import datetime, timedelta
from typing import Optional

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from logger import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = """You are RetailBrain Co-Pilot, an expert AI assistant for Sainsbury's retail operations team.
You help store managers, supply chain analysts, and buyers make faster, smarter decisions.

Your expertise covers:
- Stockout prediction and risk management
- Demand forecasting and seasonality analysis
- Replenishment planning and supplier management
- Price elasticity and promotional planning
- Anomaly detection and root-cause analysis

RESPONSE GUIDELINES:
- Be direct, specific, and action-oriented
- Use bullet points and markdown for clarity
- Always include a clear recommended action
- Quantify impacts wherever possible (units, £, % change)
- Reference the specific product IDs or categories mentioned
- Keep responses under 300 words unless deep analysis is requested
- Format numbers with commas for readability
"""


class CopilotContextBuilder:
    """Builds dynamic RAG context from current retail data."""

    def __init__(self):
        self._risk_cache: Optional[list] = None
        self._anomaly_cache: Optional[list] = None
        self._forecast_summary: Optional[list] = None

    def set_risk_data(self, risk_data: list):
        self._risk_cache = risk_data

    def set_anomaly_data(self, anomaly_data: list):
        self._anomaly_cache = anomaly_data

    def set_forecast_summary(self, forecast_summary: list):
        self._forecast_summary = forecast_summary

    def build_context(self, question: str, product_id: Optional[str] = None) -> str:
        """Build a context block to inject into the prompt."""
        sections = []
        today = datetime.now().strftime("%Y-%m-%d")
        sections.append(f"**Current Date:** {today}")

        # Risk data context
        if self._risk_cache:
            top_risks = self._risk_cache[:10]
            risk_lines = []
            for r in top_risks:
                risk_lines.append(
                    f"  - {r.get('product_name', r.get('product_id', 'Unknown'))} "
                    f"[{r.get('product_id', '')}]: risk={r.get('stockout_risk', 0):.0%}, "
                    f"stock={r.get('units_on_hand', 0):.0f}u, "
                    f"cat={r.get('category', '')}"
                )
            sections.append("**Top 10 Stockout Risks:**\n" + "\n".join(risk_lines))

        # Product-specific context
        if product_id and self._risk_cache:
            matches = [r for r in self._risk_cache if r.get("product_id") == product_id]
            if matches:
                r = matches[0]
                sections.append(
                    f"**Focus Product — {product_id}:**\n"
                    f"  Name: {r.get('product_name', 'N/A')}\n"
                    f"  Category: {r.get('category', 'N/A')}\n"
                    f"  Stockout Risk: {r.get('stockout_risk', 0):.1%}\n"
                    f"  Units on Hand: {r.get('units_on_hand', 0):.0f}\n"
                    f"  Units on Order: {r.get('units_on_order', 0):.0f}\n"
                    f"  Days of Cover: {r.get('days_of_cover', 'N/A')}\n"
                    f"  Reorder Point: {r.get('reorder_point', 0):.0f}"
                )

        # Anomaly context
        if self._anomaly_cache:
            recent_high = [a for a in self._anomaly_cache[:50] if a.get("severity") in ("high", "medium")][:5]
            if recent_high:
                anomaly_lines = []
                for a in recent_high:
                    anomaly_lines.append(
                        f"  - [{a['date']}] {a['product_id']} "
                        f"({a['anomaly_type']}, {a['severity']} severity): "
                        f"{a['root_cause']}"
                    )
                sections.append("**Recent High/Medium Anomalies:**\n" + "\n".join(anomaly_lines))

        # Forecast context
        if self._forecast_summary:
            top_forecast = self._forecast_summary[:5]
            fc_lines = [
                f"  - {f['product_id']} ({f['category']}): "
                f"{f['total_forecast_30d']:.0f} units / 30d, "
                f"vs hist: {f['vs_historical']:+.1f}%"
                for f in top_forecast
            ]
            sections.append("**Top Demand Forecasts (30-day):**\n" + "\n".join(fc_lines))

        return "\n\n".join(sections)


class RetailCopilot:
    """
    LLM Co-Pilot with RAG context injection.
    Uses OpenAI GPT-4o-mini if API key available, else deterministic fallback.
    """

    def __init__(self):
        self.context_builder = CopilotContextBuilder()
        self.conversation_history: list = []
        self.max_history = 8  # Keep last 8 turns for context
        self._openai_client = None
        self._has_openai = False
        self._init_openai()

    def _init_openai(self):
        """Initialize OpenAI client if key is available."""
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key or api_key in ("your_openai_api_key_here", "sk-...", ""):
            logger.info("No OpenAI API key found — co-pilot using rule-based fallback")
            self._has_openai = False
            return

        try:
            from openai import OpenAI  # openai>=1.0
            self._openai_client = OpenAI(api_key=api_key)
            self._has_openai = True
            logger.info("Co-Pilot: OpenAI client initialised (model: gpt-4o-mini)")
        except ImportError:
            logger.warning("openai package not installed — using rule-based fallback")
            self._has_openai = False
        except Exception as e:
            logger.warning("OpenAI init failed: %s — using rule-based fallback", e)
            self._has_openai = False

    def set_context_data(self, risk_data: list = None,
                         anomaly_data: list = None,
                         forecast_summary: list = None):
        """Inject live retail data into the co-pilot context."""
        if risk_data:
            self.context_builder.set_risk_data(risk_data)
        if anomaly_data:
            self.context_builder.set_anomaly_data(anomaly_data)
        if forecast_summary:
            self.context_builder.set_forecast_summary(forecast_summary)

    def _detect_product_id(self, question: str) -> Optional[str]:
        """Extract product ID (SAI-XXXXXX format) from question."""
        match = re.search(r"SAI-[A-Z0-9]+", question, re.IGNORECASE)
        return match.group(0).upper() if match else None

    def _classify_intent(self, question: str) -> str:
        """Classify question intent for rule-based routing."""
        q = question.lower()
        if any(w in q for w in ["risk", "stockout", "run out", "stock level"]):
            return "risk_query"
        if any(w in q for w in ["order", "replenish", "restock", "buy"]):
            return "order_recommendation"
        if any(w in q for w in ["anomaly", "spike", "crash", "unusual", "drop", "surge"]):
            return "anomaly_insight"
        if any(w in q for w in ["forecast", "predict", "next week", "expect", "demand"]):
            return "forecast_query"
        if any(w in q for w in ["category", "department", "section", "range"]):
            return "category_summary"
        if any(w in q for w in ["top", "worst", "best", "highest", "lowest", "most"]):
            return "ranking_query"
        return "general"

    def _rule_based_response(self, question: str, context: str) -> str:
        """Deterministic rule-based response when OpenAI is unavailable."""
        intent = self._classify_intent(question)
        pid = self._detect_product_id(question)

        # Extract risk data from context builder
        risk_data = self.context_builder._risk_cache or []
        anomaly_data = self.context_builder._anomaly_cache or []
        forecast_data = self.context_builder._forecast_summary or []

        if intent == "risk_query" and pid:
            matches = [r for r in risk_data if r.get("product_id") == pid]
            if matches:
                r = matches[0]
                risk_pct = r.get("stockout_risk", 0) * 100
                stock = r.get("units_on_hand", 0)
                on_order = r.get("units_on_order", 0)
                reorder = r.get("reorder_point", 0)
                action = "🚨 **Immediate action required**" if risk_pct > 70 else \
                         "⚠️ **Monitor closely**" if risk_pct > 40 else \
                         "✅ **Stock level healthy**"
                return (
                    f"## Stockout Risk Analysis: `{pid}`\n\n"
                    f"**{action}**\n\n"
                    f"| Metric | Value |\n"
                    f"|--------|-------|\n"
                    f"| Stockout Risk | **{risk_pct:.1f}%** |\n"
                    f"| Units on Hand | {stock:,.0f} units |\n"
                    f"| Units on Order | {on_order:,.0f} units |\n"
                    f"| Reorder Point | {reorder:,.0f} units |\n"
                    f"| Category | {r.get('category', 'N/A')} |\n\n"
                    f"**Recommended Action:**\n"
                    f"{'- Raise emergency purchase order for ' + str(int(max(reorder * 2 - stock - on_order, 0))) + ' units immediately.' if risk_pct > 70 else '- Review next scheduled delivery and consider pulling forward if possible.'}\n"
                    f"- Target minimum cover: 14 days of stock."
                )
            return f"Product `{pid}` not found in current risk dataset. Try searching in the Inventory Risk table."

        if intent == "order_recommendation":
            urgent = [r for r in risk_data if r.get("stockout_risk", 0) > 0.70][:5]
            if not urgent:
                return "✅ No urgent replenishment orders needed. All products are above the 70% risk threshold."
            lines = []
            for r in urgent:
                stock = r.get("units_on_hand", 0)
                reorder = r.get("reorder_point", 0)
                suggested_qty = max(int(reorder * 2 - stock - r.get("units_on_order", 0)), 0)
                name = r.get("product_name", r.get("product_id", ""))
                lines.append(
                    f"- **{name}** (`{r['product_id']}`): "
                    f"Order **{suggested_qty:,} units** · Risk: {r['stockout_risk']:.0%}"
                )
            return (
                f"## 📦 Urgent Replenishment Orders\n\n"
                f"The following {len(urgent)} products require immediate ordering:\n\n"
                + "\n".join(lines)
                + "\n\n**Total SKUs requiring action:** " + str(len(urgent))
                + "\n\n> Raise these orders in the supplier portal today to maintain 14-day cover."
            )

        if intent == "anomaly_insight":
            recent = [a for a in anomaly_data if a.get("severity") in ("high", "medium")][:5]
            if not recent:
                return "No high or medium severity anomalies detected in the recent period. Demand patterns are within normal bounds."
            lines = []
            for a in recent:
                lines.append(
                    f"- **[{a['date']}]** `{a['product_id']}` ({a['category']})\n"
                    f"  - Type: {a['anomaly_type'].replace('_', ' ').title()}\n"
                    f"  - Root Cause: {a['root_cause'].replace('_', ' ').title()}\n"
                    f"  - {a['description'][:120]}..."
                )
            return (
                f"## 🔍 Active Demand Anomalies\n\n"
                f"Found **{len(recent)}** high/medium severity anomalies:\n\n"
                + "\n\n".join(lines)
            )

        if intent == "forecast_query":
            top_fc = forecast_data[:8]
            if not top_fc:
                return "Forecast data not yet available. Ensure the forecasting engine has been run."
            lines = [
                f"- **{f['product_id']}** ({f['category']}): "
                f"{f['total_forecast_30d']:,.0f} units over 30 days "
                f"({f['vs_historical']:+.1f}% vs historical)"
                for f in top_fc
            ]
            return (
                f"## 📈 30-Day Demand Forecast\n\n"
                f"**Top products by forecasted demand:**\n\n"
                + "\n".join(lines)
                + "\n\n> Forecasts use Holt-Winters exponential smoothing with seasonal and event adjustments."
            )

        if intent == "ranking_query" and "risk" in question.lower():
            top = risk_data[:5]
            lines = [
                f"{i+1}. **{r.get('product_name', r['product_id'])}** — "
                f"Risk: **{r.get('stockout_risk', 0):.0%}** | "
                f"Stock: {r.get('units_on_hand', 0):.0f}u | "
                f"{r.get('category', '')}"
                for i, r in enumerate(top)
            ]
            return "## 🚨 Top 5 Stockout Risks\n\n" + "\n".join(lines)

        # General fallback
        return (
            "## RetailBrain Co-Pilot\n\n"
            "I can help with:\n"
            "- **Stockout risk analysis** — 'What's the risk for SAI-XXXXXX?'\n"
            "- **Replenishment orders** — 'What should I order this week?'\n"
            "- **Anomaly insights** — 'Why is demand spiking on SAI-XXXXXX?'\n"
            "- **Demand forecasts** — 'What's the forecast for frozen food?'\n"
            "- **Risk rankings** — 'Show me the top 10 stockout risks'\n\n"
            "> **Tip:** Add an OpenAI API key to `.env` to enable full AI responses."
        )

    def _openai_response(self, question: str, context: str) -> str:
        """Generate response using OpenAI GPT-4o-mini with RAG context."""
        user_message = (
            f"## Live Retail Context\n\n{context}\n\n"
            f"---\n\n"
            f"## Question\n\n{question}"
        )

        # Trim history to max_history turns
        while len(self.conversation_history) > self.max_history * 2:
            self.conversation_history.pop(0)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})

        try:
            response = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=600,
                temperature=0.3,
            )
            answer = response.choices[0].message.content.strip()
            # Save to history (without the full context injection, just question)
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            logger.warning("OpenAI request failed: %s — falling back to rule-based", e)
            return self._rule_based_response(question, context)

    def ask(self, question: str, product_id: Optional[str] = None) -> dict:
        """
        Main entry point for co-pilot questions.
        Returns dict with answer, intent, model_used, and sources used.
        """
        if not question or len(question.strip()) < 3:
            return {
                "answer": "Please ask a question about your retail operations.",
                "intent": "invalid",
                "model_used": "none",
                "product_id": None,
            }

        # Auto-detect product ID from question if not provided
        if not product_id:
            product_id = self._detect_product_id(question)

        intent = self._classify_intent(question)
        context = self.context_builder.build_context(question, product_id)

        if self._has_openai:
            answer = self._openai_response(question, context)
            model_used = "gpt-4o-mini"
        else:
            answer = self._rule_based_response(question, context)
            model_used = "rule-based"

        sources = []
        if self.context_builder._risk_cache:
            sources.append("stockout_risk_model")
        if self.context_builder._anomaly_cache:
            sources.append("anomaly_detector")
        if self.context_builder._forecast_summary:
            sources.append("demand_forecaster")

        return {
            "answer": answer,
            "intent": intent,
            "model_used": model_used,
            "product_id": product_id,
            "sources_used": sources,
        }

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


# ── Singleton accessor (for use from FastAPI) ──────────────────────────────────
_copilot_instance: Optional[RetailCopilot] = None


def get_copilot() -> RetailCopilot:
    """Return the singleton co-pilot instance."""
    global _copilot_instance
    if _copilot_instance is None:
        _copilot_instance = RetailCopilot()
    return _copilot_instance


# ── Standalone Demo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Phase 5: LLM Co-Pilot Demo")
    print("=" * 65)

    copilot = RetailCopilot()

    # Inject mock context
    copilot.set_context_data(
        risk_data=[
            {"product_id": "SAI-AB1234", "product_name": "White Hanging Heart T-Light",
             "category": "Household", "stockout_risk": 0.87,
             "units_on_hand": 12, "units_on_order": 50, "reorder_point": 100},
            {"product_id": "SAI-CD5678", "product_name": "Cream Cupid Heart Coat Hanger",
             "category": "Household", "stockout_risk": 0.72,
             "units_on_hand": 45, "units_on_order": 0, "reorder_point": 80},
        ]
    )

    questions = [
        "What should I order this week?",
        "Show me the top 5 stockout risks",
        "What's the risk for SAI-AB1234?",
        "Are there any unusual demand patterns?",
    ]

    for q in questions:
        print(f"\n🗣️  Q: {q}")
        result = copilot.ask(q)
        print(f"🤖  [Model: {result['model_used']} | Intent: {result['intent']}]")
        print(result["answer"][:300])
        print("-" * 50)

    print("\n✅ Co-Pilot demo complete.")
