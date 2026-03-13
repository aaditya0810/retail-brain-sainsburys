"""
Retail Brain × Sainsbury's — Phase 4: External Impact Factors
Integrates weather forecasts and local event data to add contextual awareness
to stockout predictions. Explains "WHY" demand spikes happen.

Weather: OpenWeatherMap API (free tier) for London forecasts.
Events: UK public events calendar (football, concerts, local markets).

Usage:
    from external_factors import get_weather_features, get_event_features, enrich_with_external
"""

import os
import hashlib
import json
import time
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

from logger import get_logger

logger = get_logger("external_factors")

# ── Configuration ──────────────────────────────────────────────────────────────
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE = "https://api.openweathermap.org/data/2.5"

# Sainsbury's London flagship coordinates
STORE_LAT = 51.5074
STORE_LON = -0.1278

# Cache directory to avoid hammering APIs
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Weather-sensitive product categories and their sensitivity profiles
WEATHER_SENSITIVITY = {
    # Hot weather (temp > 20°C) → increased demand
    "hot": {
        "Drinks":         1.35,   # +35% on hot days
        "Frozen":         1.30,   # ice cream, frozen desserts
        "Fresh Produce":  1.15,   # salads, fruits
        "Snacks":         1.10,   # BBQ snacks
    },
    # Cold weather (temp < 5°C) → increased demand
    "cold": {
        "Ambient Grocery": 1.25,  # soups, stews, comfort food
        "Fresh Bakery":    1.15,  # warm bread, pastries
        "Meat & Fish":     1.20,  # roasts, stews
        "Household":       1.05,
    },
    # Rain → increased demand for indoor/comfort items
    "rain": {
        "Ambient Grocery": 1.15,  # comfort food
        "Frozen":          1.20,  # ready meals
        "Snacks":          1.15,  # stay-at-home snacking
        "Drinks":          1.10,  # hot drinks ingredients
    },
}

# Product-level weather sensitivity overrides (specific SKU patterns)
WEATHER_PRODUCT_KEYWORDS = {
    "hot": {
        "ice cream": 1.60,
        "lemonade": 1.45,
        "water": 1.40,
        "cola": 1.30,
        "strawberries": 1.35,
        "salad": 1.30,
        "bbq": 1.50,
        "sausage": 1.35,   # BBQ sausages
    },
    "cold": {
        "soup": 1.50,
        "bread": 1.20,
        "porridge": 1.35,
        "oats": 1.30,
        "tea": 1.25,
        "coffee": 1.20,
        "butter": 1.15,
    },
    "rain": {
        "umbrella": 1.80,
        "baked beans": 1.25,
        "pizza": 1.30,
        "lasagne": 1.30,
        "chips": 1.20,
    },
}

# ── UK Local Events Database ──────────────────────────────────────────────────
# Major recurring London events that affect Sainsbury's footfall
UK_LOCAL_EVENTS = {
    # Football match days — London Premier League fixtures (simplified Q4 2024)
    "premier_league": {
        "impact_radius_km": 5,
        "footfall_multiplier": 1.25,
        "affected_categories": ["Drinks", "Snacks", "Meat & Fish"],
        "dates": [
            # Arsenal, Chelsea, Tottenham, West Ham home matches (sample dates)
            "2024-10-05", "2024-10-19", "2024-10-26",
            "2024-11-02", "2024-11-09", "2024-11-23", "2024-11-30",
            "2024-12-04", "2024-12-14", "2024-12-21", "2024-12-26",
        ],
    },
    # London Marathon / Winter events
    "winter_wonderland": {
        "impact_radius_km": 10,
        "footfall_multiplier": 1.15,
        "affected_categories": ["Drinks", "Snacks", "Fresh Bakery"],
        "dates": [str(date(2024, 11, 17) + timedelta(days=i)) for i in range(45)],
    },
    # New Year's Eve
    "nye_fireworks": {
        "impact_radius_km": 15,
        "footfall_multiplier": 1.50,
        "affected_categories": ["Drinks", "Snacks", "Fresh Produce"],
        "dates": ["2024-12-30", "2024-12-31"],
    },
    # School half term (Oct)
    "school_half_term": {
        "impact_radius_km": 20,
        "footfall_multiplier": 1.10,
        "affected_categories": ["Snacks", "Drinks", "Frozen", "Fresh Bakery"],
        "dates": [str(date(2024, 10, 28) + timedelta(days=i)) for i in range(5)],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# WEATHER DATA INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def _cache_key(prefix: str, params: dict) -> str:
    """Generate a deterministic cache key."""
    raw = f"{prefix}:{json.dumps(params, sort_keys=True)}"
    return hashlib.md5(raw.encode()).hexdigest()


def _read_cache(key: str, max_age_seconds: int = 3600) -> Optional[dict]:
    """Read from file cache if not expired."""
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            cached = json.load(f)
        if time.time() - cached.get("_cached_at", 0) > max_age_seconds:
            return None
        return cached.get("data")
    except (json.JSONDecodeError, KeyError):
        return None


def _write_cache(key: str, data: dict):
    """Write data to file cache."""
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump({"_cached_at": time.time(), "data": data}, f)


def fetch_weather_forecast(lat: float = STORE_LAT, lon: float = STORE_LON,
                           days: int = 5) -> Optional[list]:
    """
    Fetch weather forecast from OpenWeatherMap API.
    Returns a list of daily forecasts with temp, rain, wind, condition.
    Falls back to synthetic data if API key not configured.
    """
    cache_key = _cache_key("weather", {"lat": lat, "lon": lon, "days": days})
    cached = _read_cache(cache_key, max_age_seconds=3600)
    if cached:
        logger.info("Using cached weather forecast")
        return cached

    if OPENWEATHER_API_KEY and OPENWEATHER_API_KEY != "your_openweather_api_key_here":
        try:
            resp = requests.get(
                f"{OPENWEATHER_BASE}/forecast",
                params={
                    "lat": lat, "lon": lon,
                    "appid": OPENWEATHER_API_KEY,
                    "units": "metric",
                    "cnt": days * 8,  # 3-hour intervals
                },
                timeout=10,
            )
            resp.raise_for_status()
            raw = resp.json()

            # Aggregate 3-hour intervals into daily summaries
            daily = {}
            for item in raw.get("list", []):
                dt = datetime.fromtimestamp(item["dt"]).date().isoformat()
                if dt not in daily:
                    daily[dt] = {
                        "date": dt,
                        "temps": [],
                        "rain_mm": 0.0,
                        "conditions": [],
                        "wind_speeds": [],
                    }
                daily[dt]["temps"].append(item["main"]["temp"])
                daily[dt]["wind_speeds"].append(item["wind"]["speed"])
                daily[dt]["conditions"].append(item["weather"][0]["main"])
                if "rain" in item:
                    daily[dt]["rain_mm"] += item["rain"].get("3h", 0.0)

            forecasts = []
            for dt, d in sorted(daily.items()):
                forecasts.append({
                    "date": dt,
                    "temp_avg": round(np.mean(d["temps"]), 1),
                    "temp_max": round(max(d["temps"]), 1),
                    "temp_min": round(min(d["temps"]), 1),
                    "rain_mm": round(d["rain_mm"], 1),
                    "wind_avg": round(np.mean(d["wind_speeds"]), 1),
                    "condition": max(set(d["conditions"]), key=d["conditions"].count),
                    "is_rainy": d["rain_mm"] > 2.0,
                })

            _write_cache(cache_key, forecasts)
            logger.info(f"Fetched {len(forecasts)}-day weather forecast from API")
            return forecasts

        except requests.exceptions.RequestException as e:
            logger.warning(f"Weather API failed: {e}. Using synthetic fallback.")

    # ── Synthetic fallback (UK Q4 averages for London) ────────────────────────
    logger.info("No weather API key — generating synthetic UK Q4 weather")
    return _generate_synthetic_weather(days)


def _generate_synthetic_weather(days: int = 7) -> list:
    """Generate realistic UK Q4 weather for London."""
    today = date.today()
    forecasts = []
    for i in range(days):
        d = today + timedelta(days=i)
        month = d.month

        # UK Q4 temperature ranges (°C)
        if month == 10:
            temp_avg = np.random.normal(12.0, 3.0)
        elif month == 11:
            temp_avg = np.random.normal(8.0, 3.0)
        else:  # December
            temp_avg = np.random.normal(5.0, 3.5)

        rain_mm = max(0, np.random.exponential(3.0))
        is_rainy = rain_mm > 2.0

        if temp_avg > 15:
            condition = "Clear" if not is_rainy else "Rain"
        elif temp_avg > 5:
            condition = np.random.choice(["Clouds", "Rain", "Drizzle"],
                                          p=[0.4, 0.35, 0.25])
        else:
            condition = np.random.choice(["Clouds", "Rain", "Snow", "Fog"],
                                          p=[0.3, 0.3, 0.2, 0.2])

        forecasts.append({
            "date": d.isoformat(),
            "temp_avg": round(temp_avg, 1),
            "temp_max": round(temp_avg + np.random.uniform(2, 5), 1),
            "temp_min": round(temp_avg - np.random.uniform(2, 4), 1),
            "rain_mm": round(rain_mm, 1),
            "wind_avg": round(np.random.uniform(5, 25), 1),
            "condition": condition,
            "is_rainy": is_rainy,
        })

    return forecasts


def compute_weather_multiplier(weather: dict, category: str,
                                product_name: str = "") -> float:
    """
    Calculate a demand multiplier based on weather conditions for a specific
    product category and optional product name.

    Returns a float multiplier (1.0 = no effect, >1 = increased demand).
    """
    temp = weather.get("temp_avg", 10.0)
    is_rainy = weather.get("is_rainy", False)
    multiplier = 1.0

    # Temperature effects
    if temp > 20:
        cat_mult = WEATHER_SENSITIVITY["hot"].get(category, 1.0)
        multiplier *= cat_mult
    elif temp < 5:
        cat_mult = WEATHER_SENSITIVITY["cold"].get(category, 1.0)
        multiplier *= cat_mult

    # Rain effects (compound with temperature)
    if is_rainy:
        rain_mult = WEATHER_SENSITIVITY["rain"].get(category, 1.0)
        multiplier *= rain_mult

    # Product-specific keyword matching
    name_lower = product_name.lower()
    weather_type = "hot" if temp > 20 else ("cold" if temp < 5 else "rain" if is_rainy else None)
    if weather_type and weather_type in WEATHER_PRODUCT_KEYWORDS:
        for keyword, kw_mult in WEATHER_PRODUCT_KEYWORDS[weather_type].items():
            if keyword in name_lower:
                multiplier = max(multiplier, kw_mult)
                break

    return round(multiplier, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL EVENT DATA INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_local_events(target_date: str) -> list:
    """
    Return all local events happening on a given date.
    Each event dict includes: name, footfall_multiplier, affected_categories.
    """
    events = []
    for event_name, event_data in UK_LOCAL_EVENTS.items():
        if target_date in event_data["dates"]:
            events.append({
                "event_name": event_name,
                "footfall_multiplier": event_data["footfall_multiplier"],
                "affected_categories": event_data["affected_categories"],
            })
    return events


def compute_event_multiplier(target_date: str, category: str) -> float:
    """
    Calculate a demand multiplier from local events for a category on a given date.
    Multiple events compound multiplicatively.
    """
    events = get_local_events(target_date)
    multiplier = 1.0
    for event in events:
        if category in event["affected_categories"]:
            multiplier *= event["footfall_multiplier"]
    return round(multiplier, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════

def enrich_with_external(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a product DataFrame with weather and event impact factors.
    Adds columns: weather_multiplier, event_multiplier_local, external_demand_factor,
                  weather_condition, weather_temp, weather_rain_mm.

    These feed directly into the upgraded feature engineering pipeline.
    """
    result = df.copy()

    # Fetch weather forecast (covers next 5 days)
    forecasts = fetch_weather_forecast()
    weather_by_date = {}
    if forecasts:
        for f in forecasts:
            weather_by_date[f["date"]] = f

    weather_mults = []
    event_mults = []
    conditions = []
    temps = []
    rain_mms = []

    for _, row in result.iterrows():
        row_date = str(row.get("date", ""))[:10]
        category = row.get("category", "")
        product_name = row.get("product_name", "")

        # Weather
        weather = weather_by_date.get(row_date, {})
        w_mult = compute_weather_multiplier(weather, category, product_name)
        weather_mults.append(w_mult)
        conditions.append(weather.get("condition", "Unknown"))
        temps.append(weather.get("temp_avg", None))
        rain_mms.append(weather.get("rain_mm", None))

        # Events
        e_mult = compute_event_multiplier(row_date, category)
        event_mults.append(e_mult)

    result["weather_multiplier"] = weather_mults
    result["local_event_multiplier"] = event_mults
    result["external_demand_factor"] = (
        result["weather_multiplier"] * result["local_event_multiplier"]
    )
    result["weather_condition"] = conditions
    result["weather_temp"] = temps
    result["weather_rain_mm"] = rain_mms

    logger.info(
        f"Enriched {len(result)} rows with external factors. "
        f"Avg weather mult: {result['weather_multiplier'].mean():.3f}, "
        f"Avg event mult: {result['local_event_multiplier'].mean():.3f}"
    )

    return result


def get_weather_impact_summary(forecasts: list = None) -> list:
    """
    Generate a human-readable weather impact summary for the dashboard.
    Returns a list of dicts with date, condition, temp, and affected categories.
    """
    if forecasts is None:
        forecasts = fetch_weather_forecast()
    if not forecasts:
        return []

    summaries = []
    for f in forecasts:
        impacts = []
        temp = f["temp_avg"]
        if temp > 20:
            impacts.append({
                "trigger": f"Hot weather ({temp}°C)",
                "categories": list(WEATHER_SENSITIVITY["hot"].keys()),
                "direction": "increase",
            })
        elif temp < 5:
            impacts.append({
                "trigger": f"Cold weather ({temp}°C)",
                "categories": list(WEATHER_SENSITIVITY["cold"].keys()),
                "direction": "increase",
            })
        if f.get("is_rainy"):
            impacts.append({
                "trigger": f"Rain ({f['rain_mm']}mm)",
                "categories": list(WEATHER_SENSITIVITY["rain"].keys()),
                "direction": "increase",
            })

        # Local events
        events = get_local_events(f["date"])
        for event in events:
            impacts.append({
                "trigger": event["event_name"].replace("_", " ").title(),
                "categories": event["affected_categories"],
                "direction": "increase",
            })

        summaries.append({
            "date": f["date"],
            "condition": f["condition"],
            "temp_avg": f["temp_avg"],
            "rain_mm": f["rain_mm"],
            "impacts": impacts,
        })

    return summaries


if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 4: External Impact Factors")
    print("=" * 60)

    # Weather
    print("\n🌤️  Weather Forecast:")
    forecasts = fetch_weather_forecast()
    if forecasts:
        for f in forecasts:
            print(f"  {f['date']}: {f['condition']} {f['temp_avg']}°C, rain: {f['rain_mm']}mm")

    # Events
    print("\n📅  Local Events Check:")
    test_dates = ["2024-10-31", "2024-11-29", "2024-12-24", "2024-12-31"]
    for d in test_dates:
        events = get_local_events(d)
        if events:
            names = [e["event_name"] for e in events]
            print(f"  {d}: {', '.join(names)}")

    # Weather impact summary
    print("\n📊  Weather Impact Summary:")
    summary = get_weather_impact_summary(forecasts)
    for s in summary[:3]:
        print(f"  {s['date']}: {s['condition']} {s['temp_avg']}°C")
        for imp in s["impacts"]:
            print(f"    → {imp['trigger']}: {', '.join(imp['categories'])} ({imp['direction']})")
