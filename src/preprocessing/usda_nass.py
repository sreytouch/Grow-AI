"""
GrowAI Phase 2 — USDA NASS Quick Stats Pre-Processing
=======================================================
Authors: Madike Asia, Sreytouch Lang (Jessica)
Thullen Lab, July 2026

Pipeline:
  1. Fetch crop yield data via USDA NASS API (free key required)
  2. Filter to home-garden-relevant commodities
  3. Clean: drop missing/withheld values, standardize columns
  4. Feature engineer: 3-year rolling avg, yield trend
  5. Normalize: MinMaxScaler on all continuous features
  6. Export: CSV + Parquet

API docs : https://quickstats.nass.usda.gov/api
Free key : https://quickstats.nass.usda.gov/api (register → get key instantly)

Reference:
  USDA National Agricultural Statistics Service. Quick Stats 2.0 API.
  Scientific Reports (Jul 2025) — Crop recommendation with XGBoost/RF.
  MixINN arXiv:2605.04744 — >30% missing column drop protocol.
"""

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.utils.config import (
    USDA_API_KEY, USDA_BASE_URL, HOME_GARDEN_CROPS,
    USDA_START_YEAR, MISSING_COL_THRESHOLD,
    USDA_CLEAN_CSV, USDA_CLEAN_PARQUET,
)


# ── Step 1: Fetch from API ────────────────────────────────────────────────────

def fetch_usda_nass(commodity: str, api_key: str = USDA_API_KEY) -> pd.DataFrame:
    """
    Fetch USDA NASS yield data for a single commodity.

    Parameters
    ----------
    commodity : USDA commodity name (e.g. 'TOMATOES')
    api_key   : your registered USDA Quick Stats API key

    Returns
    -------
    DataFrame with raw API response data
    """
    params = {
        "key":               api_key,
        "commodity_desc":    commodity,
        "statisticcat_desc": "YIELD",
        "year__GE":          str(USDA_START_YEAR),
        "format":            "JSON",
    }
    try:
        response = requests.get(USDA_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json().get("data", [])
        df = pd.DataFrame(data)
        print(f"  Fetched {len(df):,} rows for {commodity}")
        return df
    except requests.RequestException as e:
        print(f"  [ERROR] Failed to fetch {commodity}: {e}")
        return pd.DataFrame()


def fetch_all_crops(api_key: str = USDA_API_KEY) -> pd.DataFrame:
    """Fetch and concatenate data for all home-garden crops."""
    print("\n=== Fetching USDA NASS data ===")
    frames = [fetch_usda_nass(crop, api_key) for crop in HOME_GARDEN_CROPS]
    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    print(f"Combined raw rows: {len(combined):,}")
    return combined


# ── Step 2–6: Full pre-processing ────────────────────────────────────────────

def preprocess_usda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pre-processing pipeline for raw USDA NASS DataFrame.

    Steps
    -----
    1. Column selection and name standardization
    2. Drop rows missing value or state_name
    3. Remove USDA withheld values '(D)'
    4. Drop columns with >30% missing
    5. Type conversion
    6. Feature engineering (3-yr rolling avg, yield trend)
    7. MinMaxScaler normalization

    Returns cleaned DataFrame ready for modeling.
    """
    print(f"\nRaw shape: {df.shape}")

    # 1. Keep relevant columns
    keep = ["year", "state_name", "commodity_desc",
            "statisticcat_desc", "unit_desc", "Value"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Standardize column names → snake_case
    df.columns = (
        df.columns.str.lower()
        .str.replace(r"[ \(\)%]", "_", regex=True)
        .str.strip("_")
    )

    # 2. Drop rows missing value or state
    df = df.dropna(subset=["value", "state_name"])

    # 3. Remove USDA withheld values
    df = df[~df["value"].astype(str).str.strip().isin(["(D)", "(Z)", "(NA)"])]

    # 4. Drop columns with > threshold missing
    thresh = int((1 - MISSING_COL_THRESHOLD) * len(df))
    df = df.dropna(thresh=thresh, axis=1)

    # 5. Type conversion
    df["value"] = pd.to_numeric(
        df["value"].astype(str).str.replace(",", ""), errors="coerce"
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["value", "year"])
    df["year"] = df["year"].astype(int)
    df["state_name"]    = df["state_name"].astype("category")
    df["commodity_desc"] = df["commodity_desc"].astype("category")

    # 6. Feature engineering
    df = df.sort_values(["commodity_desc", "state_name", "year"])
    grp = df.groupby(["commodity_desc", "state_name"])["value"]
    df["yield_3yr_avg"] = grp.transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df["yield_trend"] = grp.transform(
        lambda x: x.diff().rolling(3, min_periods=1).mean()
    )

    # 7. Normalize
    scaler = MinMaxScaler()
    num_cols = ["value", "yield_3yr_avg", "yield_trend"]
    num_cols = [c for c in num_cols if c in df.columns]
    df[[f"{c}_norm" for c in num_cols]] = scaler.fit_transform(
        df[num_cols].fillna(0)
    )

    print(f"Processed shape: {df.shape}")
    return df.reset_index(drop=True)


# ── Export ────────────────────────────────────────────────────────────────────

def export(df: pd.DataFrame):
    USDA_CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(USDA_CLEAN_CSV, index=False)
    df.to_parquet(USDA_CLEAN_PARQUET, index=False)
    print(f"Saved → {USDA_CLEAN_CSV}")
    print(f"Saved → {USDA_CLEAN_PARQUET}")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(api_key: str = USDA_API_KEY) -> pd.DataFrame:
    print("\n=== USDA NASS Pre-Processing Pipeline ===")
    if api_key == "YOUR_USDA_API_KEY_HERE":
        print("[ERROR] Set USDA_API_KEY in your environment or src/utils/config.py first.")
        return pd.DataFrame()

    raw = fetch_all_crops(api_key)
    if raw.empty:
        print("[ERROR] No data fetched. Check your API key and internet connection.")
        return pd.DataFrame()

    clean = preprocess_usda(raw)
    export(clean)
    print("\n=== USDA NASS Pipeline Complete ===")
    return clean


if __name__ == "__main__":
    df = run_pipeline()
    if not df.empty:
        print(df.head())
