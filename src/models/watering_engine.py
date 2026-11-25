"""
GrowAI Phase 2 — Watering Rule Engine
=======================================
Authors: Madike Asia, Sreytouch Lang (Jessica)
Thullen Lab, July 2026

Rule-based system using FAO-56 Penman-Monteith ET₀.
No model training required — transparent and explainable.

Formula: water_deficit = max(ET₀ − precipitation, 0)
         water_needed  = water_deficit > crop_threshold

Crop-specific thresholds (mm/day) are based on FAO-56 irrigation guidelines.

Reference:
  FAO (1998). Crop evapotranspiration. Irrigation & Drainage Paper 56.
  Frontiers Sustainable Food Systems (Feb 2026). doi:10.3389/fsufs.2026.1734366
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import date

from src.utils.config import WEATHER_CACHE_DB, DEFAULT_LAT, DEFAULT_LON

# Crop-specific daily water deficit thresholds (mm/day)
# Based on FAO-56 crop coefficient guidance
CROP_THRESHOLDS = {
    "tomato":     4.0,
    "lettuce":    3.0,
    "pepper":     4.5,
    "strawberry": 3.5,
    "cucumber":   5.0,
    "squash":     5.0,
    "basil":      3.0,
    "mint":       3.5,
    "herbs":      3.0,
    "default":    5.0,   # fallback
}


def get_watering_schedule(lat: float = DEFAULT_LAT,
                          lon: float = DEFAULT_LON,
                          crop: str = "default",
                          days: int = 7) -> pd.DataFrame:
    """
    Generate a watering schedule for the next N days.

    Parameters
    ----------
    lat, lon : location coordinates
    crop     : crop type (used to look up threshold)
    days     : forecast horizon in days

    Returns
    -------
    DataFrame with date, water_deficit_mm, water_needed, recommended_amount_L
    """
    table = f"weather_{str(lat).replace('.','_')}_{str(lon).replace('.','_').replace('-','n')}"
    threshold = CROP_THRESHOLDS.get(crop.lower(), CROP_THRESHOLDS["default"])

    if not WEATHER_CACHE_DB.exists():
        raise FileNotFoundError(
            "Weather cache not found. Run src/preprocessing/open_meteo.py first."
        )

    with sqlite3.connect(str(WEATHER_CACHE_DB)) as conn:
        try:
            df = pd.read_sql(f"SELECT * FROM '{table}'", conn)
        except Exception:
            raise ValueError(
                f"No weather data for ({lat}, {lon}). "
                "Run open_meteo.py with that location first."
            )

    df["time"] = pd.to_datetime(df["time"])
    df["et0"]  = pd.to_numeric(df["et0"], errors="coerce").fillna(0)
    df["precip"] = pd.to_numeric(df["precip"], errors="coerce").fillna(0)
    df["water_deficit_mm"] = (df["et0"] - df["precip"]).clip(lower=0)
    df["water_needed"] = df["water_deficit_mm"] > threshold

    # Estimate recommended watering amount (liters per m²)
    # Approximation: 1 mm deficit ≈ 1 L/m²
    df["recommended_amount_L_per_m2"] = df["water_deficit_mm"].where(
        df["water_needed"], other=0.0
    ).round(1)

    # Return only forecast days
    today = pd.Timestamp(date.today())
    schedule = (
        df[df["time"] >= today]
        [["time", "et0", "precip", "water_deficit_mm", "water_needed", "recommended_amount_L_per_m2"]]
        .head(days)
        .reset_index(drop=True)
    )

    print(f"\nWatering schedule for {crop} at ({lat}, {lon}) — threshold={threshold} mm/day")
    print(schedule.to_string(index=False))
    return schedule


def watering_summary(schedule: pd.DataFrame) -> str:
    """Generate a plain-language watering summary for the chatbot."""
    water_days = schedule[schedule["water_needed"]]
    if water_days.empty:
        return "No watering needed in the next forecast period. Precipitation is sufficient."

    lines = ["Based on today's ET₀ and rainfall forecast, water your plants on:"]
    for _, row in water_days.iterrows():
        day_str = row["time"].strftime("%A, %b %d")
        lines.append(
            f"  • {day_str} — apply {row['recommended_amount_L_per_m2']:.1f} L/m²"
            f" (deficit: {row['water_deficit_mm']:.1f} mm)"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    schedule = get_watering_schedule(crop="tomato")
    print("\n" + watering_summary(schedule))
