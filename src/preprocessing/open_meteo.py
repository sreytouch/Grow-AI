"""
GrowAI Phase 2 — Open-Meteo Weather API Pre-Processing
========================================================
Authors: Madike Asia, Sreytouch Lang (Jessica)
Thullen Lab, July 2026

Pipeline:
  1. Fetch 90-day historical + 7-day forecast via Open-Meteo REST API
  2. Parse: temperature, precipitation, ET₀ (FAO-56 Penman-Monteith)
  3. Handle missing values with forward-fill (≤2 days)
  4. Engineer water_deficit = max(ET₀ − P, 0) and water_needed flag
  5. Compute 7-day rolling averages for trend features
  6. Cache per-location in SQLite

No API key required for non-commercial use (CC BY 4.0).
API docs: https://open-meteo.com/en/docs

Reference:
  FAO (1998). Crop Evapotranspiration. Irrigation & Drainage Paper 56.
  Frontiers Sustainable Food Systems (Feb 2026). doi:10.3389/fsufs.2026.1734366
  PMC Smart Irrigation Survey (2022). Forward-fill ≤2 days protocol.
"""

import argparse
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import requests

from src.utils.config import (
    DATA_PROCESSED, HISTORY_DAYS, FORECAST_DAYS,
    WATERING_THRESHOLD_MM, FFILL_LIMIT,
    DEFAULT_LAT, DEFAULT_LON, WEATHER_CACHE_DB,
)

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "uv_index_max",
]

REQUIRED_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
]


# ── Step 1: Fetch ─────────────────────────────────────────────────────────────

def fetch_weather(lat: float = DEFAULT_LAT,
                  lon: float = DEFAULT_LON,
                  history_days: int = HISTORY_DAYS,
                  forecast_days: int = FORECAST_DAYS) -> pd.DataFrame:
    """
    Fetch historical + forecast weather for a location.

    Parameters
    ----------
    lat, lon      : decimal degrees (e.g. 33.77, -118.19 for Long Beach CA)
    history_days  : days of historical data to fetch
    forecast_days : days of forecast to append

    Returns
    -------
    Combined DataFrame with daily weather variables
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=history_days)

    hist_params = {
        "latitude":   lat, "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date":   end_date.strftime("%Y-%m-%d"),
        "daily":      ",".join(DAILY_VARS),
        "timezone":   "auto",
    }
    fcast_params = {
        "latitude":      lat, "longitude": lon,
        "daily":         ",".join(DAILY_VARS),
        "forecast_days": forecast_days,
        "timezone":      "auto",
    }

    print(f"Fetching weather for ({lat}, {lon}) ...")
    try:
        hist_response = requests.get(ARCHIVE_URL, params=hist_params, timeout=30)
        hist_response.raise_for_status()
        fcast_response = requests.get(FORECAST_URL, params=fcast_params, timeout=30)
        fcast_response.raise_for_status()

        hist = pd.DataFrame(hist_response.json().get("daily", {}))
        fcast = pd.DataFrame(fcast_response.json().get("daily", {}))
        combined = pd.concat([hist, fcast], ignore_index=True)
        combined["lat"] = lat
        combined["lon"] = lon
        print(f"  Fetched {len(combined)} days ({start_date.date()} → +{forecast_days} forecast)")
        return combined
    except requests.RequestException as e:
        print(f"[ERROR] Weather fetch failed: {e}")
        return pd.DataFrame()


# ── Step 2–5: Pre-process ────────────────────────────────────────────────────

def preprocess_weather(df: pd.DataFrame,
                       threshold: float = WATERING_THRESHOLD_MM,
                       ffill_limit: int  = FFILL_LIMIT) -> pd.DataFrame:
    """
    Clean and engineer features from raw Open-Meteo data.

    Parameters
    ----------
    df        : raw DataFrame from fetch_weather()
    threshold : water deficit (mm/day) above which water_needed = True
    ffill_limit : max consecutive missing days to forward-fill

    Returns
    -------
    Cleaned DataFrame with ET₀ deficit and water_needed flag
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    duplicate_days = int(df["time"].duplicated(keep="last").sum())
    if duplicate_days:
        print(f"  Dropping {duplicate_days} overlapping archive/forecast day(s)")
    df = (
        df.sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .reset_index(drop=True)
    )

    # Forward-fill missing values (≤ limit consecutive days)
    numeric_cols = [col for col in DAILY_VARS if col in df.columns]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].ffill(limit=ffill_limit)

    gap_cols = [col for col in REQUIRED_DAILY_VARS if col in df.columns]
    if gap_cols:
        df["data_gap_flag"] = df[gap_cols].isnull().any(axis=1)
    else:
        df["data_gap_flag"] = False

    # ET₀ and precipitation shorthand
    df["et0"]   = df.get("et0_fao_evapotranspiration", pd.Series(0, index=df.index))
    df["precip"] = df.get("precipitation_sum",          pd.Series(0, index=df.index))

    # Water deficit (FAO-56 formula)
    df["water_deficit_mm"] = (df["et0"] - df["precip"]).clip(lower=0)
    df["water_needed"]     = df["water_deficit_mm"] > threshold

    # 7-day rolling averages
    for col in ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"]:
        if col in df.columns:
            df[f"{col}_7d_avg"] = df[col].rolling(7, min_periods=1).mean()

    days_needing_water = int(df["water_needed"].sum())
    print(f"  Date range    : {df['time'].min().date()} → {df['time'].max().date()}")
    print(f"  Days needing water: {days_needing_water} / {len(df)}")
    print(f"  Data gaps flagged : {int(df['data_gap_flag'].sum())}")

    return df


# ── Step 6: Cache to SQLite ───────────────────────────────────────────────────

def save_to_cache(df: pd.DataFrame, lat: float, lon: float):
    """Save processed weather data to SQLite cache, keyed by location."""
    WEATHER_CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    table = f"weather_{str(lat).replace('.','_')}_{str(lon).replace('.','_').replace('-','n')}"
    with sqlite3.connect(str(WEATHER_CACHE_DB)) as conn:
        df.astype(str).to_sql(table, conn, if_exists="replace", index=False)
    print(f"  Cached → {WEATHER_CACHE_DB} (table: {table})")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(lat: float = DEFAULT_LAT,
                 lon: float = DEFAULT_LON,
                 save_csv: bool = True) -> pd.DataFrame:
    print("\n=== Open-Meteo Weather Pre-Processing Pipeline ===")
    raw   = fetch_weather(lat, lon)
    if raw.empty:
        return raw
    clean = preprocess_weather(raw)
    save_to_cache(clean, lat, lon)

    if save_csv:
        out_path = DATA_PROCESSED / f"weather_{lat}_{lon}.csv"
        clean.to_csv(out_path, index=False)
        print(f"  CSV saved → {out_path}")

    print("\n=== Open-Meteo Pipeline Complete ===")
    return clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and pre-process Open-Meteo weather data")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Latitude")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Longitude")
    args = parser.parse_args()
    df = run_pipeline(lat=args.lat, lon=args.lon)
    if not df.empty:
        print(df[["time", "et0", "precip", "water_deficit_mm", "water_needed"]].tail(10))
