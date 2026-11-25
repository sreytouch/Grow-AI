"""
GrowAI Phase 2 — Kaggle Agriculture & Farming Pre-Processing
=============================================================
Authors: Madike Asia, Sreytouch Lang (Jessica)
Thullen Lab, July 2026

Pipeline:
  1. License audit (manual step — see README note)
  2. Load and filter to home-garden-relevant crops
  3. Drop columns with >40% missing; impute numerics with median
  4. Standardize crop labels to match USDA NASS commodity names
  5. Engineer cost_per_yield ratio
  6. Merge with cleaned USDA NASS table on crop name key
  7. Export merged CSV

Download:
  kaggle datasets download -d atharvaingle/crop-recommendation-dataset

Reference:
  arXiv:2410.08295 — missing value imputation best practices.
  Heliyon (Jan 2024) — NPK crop recommendation preprocessing.
  Scientific Reports (Jul 2025) — XGBoost/RF feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import (
    DATA_RAW, DATA_PROCESSED, MISSING_COL_THRESHOLD,
    KAGGLE_CLEAN_CSV, USDA_CLEAN_CSV,
)

# Home-garden crop label mapping (Kaggle names → USDA NASS commodity names)
CROP_NAME_MAP = {
    "rice":      None,          # not relevant to home gardens — exclude
    "maize":     "CORN",
    "chickpea":  None,
    "kidneybeans": "BEANS",
    "pigeonpeas": None,
    "mothbeans": None,
    "mungbean":  "BEANS",
    "blackgram": None,
    "lentil":    None,
    "pomegranate": None,
    "banana":    None,
    "mango":     None,
    "grapes":    "GRAPES",
    "watermelon": None,
    "muskmelon": None,
    "apple":     "APPLES",
    "orange":    None,
    "papaya":    None,
    "coconut":   None,
    "cotton":    None,
    "jute":      None,
    "coffee":    None,
    "tomato":    "TOMATOES",
    "pepper":    "PEPPERS",
    "lettuce":   "LETTUCE",
    "beans":     "BEANS",
    "cucumber":  "CUCUMBERS",
    "squash":    "SQUASH",
    "strawberry": "STRAWBERRIES",
    "herbs":     "HERBS",
    "potato":    "POTATOES",
    "carrot":    None,
    "spinach":   None,
    "kale":      None,
}

HOME_GARDEN_USDA_NAMES = {v for v in CROP_NAME_MAP.values() if v is not None}


def preprocess_kaggle_agri(filepath: str | Path = None) -> pd.DataFrame:
    """
    Full pre-processing pipeline for Kaggle crop recommendation CSV.

    Parameters
    ----------
    filepath : path to Crop_recommendation.csv (defaults to data/raw/)

    Returns
    -------
    Cleaned DataFrame ready for crop recommendation modeling
    """
    if filepath is None:
        filepath = DATA_RAW / "Crop_recommendation.csv"

    print("\n=== Kaggle Agriculture Pre-Processing Pipeline ===")
    df = pd.read_csv(filepath)
    print(f"Raw shape: {df.shape}")
    print(f"Columns  : {list(df.columns)}")

    # 1. Standardize column names
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # 2. Drop columns with >40% missing
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.40].index.tolist()
    if high_missing:
        print(f"Dropping {len(high_missing)} columns with >40% missing: {high_missing}")
        df = df.drop(columns=high_missing)

    # 3. Impute remaining numerics with median
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # 4. Standardize crop label column
    label_col = next((c for c in df.columns if "label" in c or "crop" in c), None)
    if label_col:
        df[label_col] = df[label_col].str.lower().str.strip()

        # Map to USDA names and filter to home-garden crops
        df["usda_commodity"] = df[label_col].map(CROP_NAME_MAP)
        df_filtered = df[df["usda_commodity"].notna()].copy()
        print(f"After home-garden filter: {len(df_filtered):,} rows "
              f"({len(df_filtered[label_col].unique())} crops)")
    else:
        print("[WARNING] No label column found. Using full dataset.")
        df_filtered = df.copy()
        df_filtered["usda_commodity"] = None

    # 5. Engineer cost_per_yield if applicable columns exist
    if "cost" in df_filtered.columns and "yield" in df_filtered.columns:
        df_filtered["cost_per_yield"] = (
            df_filtered["cost"] / (df_filtered["yield"] + 1e-9)
        )
        print("Engineered: cost_per_yield")

    print(f"Final shape: {df_filtered.shape}")
    return df_filtered.reset_index(drop=True)


def merge_with_usda(kaggle_df: pd.DataFrame,
                    usda_path: Path = USDA_CLEAN_CSV) -> pd.DataFrame:
    """
    Merge Kaggle features with USDA NASS yield data on commodity name.

    Returns merged DataFrame, or kaggle_df alone if USDA file not found.
    """
    if not usda_path.exists():
        print(f"[WARNING] USDA file not found at {usda_path}. Skipping merge.")
        return kaggle_df

    usda = pd.read_csv(usda_path)
    usda["commodity_desc"] = usda["commodity_desc"].astype(str).str.upper()
    kaggle_df["usda_commodity"] = kaggle_df["usda_commodity"].astype(str).str.upper()
    usda_features = (
        usda.groupby("commodity_desc", as_index=False)[["yield_3yr_avg", "yield_trend"]]
        .mean()
    )

    merged = kaggle_df.merge(
        usda_features,
        left_on="usda_commodity",
        right_on="commodity_desc",
        how="left",
    )
    matched_rows = int(merged["yield_3yr_avg"].notna().sum())
    print(f"After USDA merge: {merged.shape} | matched rows: {matched_rows:,}/{len(merged):,}")
    return merged


def run_pipeline(filepath: str | Path = None) -> pd.DataFrame:
    clean  = preprocess_kaggle_agri(filepath)
    merged = merge_with_usda(clean)

    KAGGLE_CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(KAGGLE_CLEAN_CSV, index=False)
    print(f"Saved → {KAGGLE_CLEAN_CSV}")
    print("\n=== Kaggle Agriculture Pipeline Complete ===")
    return merged


if __name__ == "__main__":
    df = run_pipeline()
    if not df.empty:
        print(df.head())
