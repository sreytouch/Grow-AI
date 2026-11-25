"""
GrowAI Phase 2 — Shared Configuration
======================================
Authors: Madike Asia, Sreytouch Lang (Jessica)
Thullen Lab, July 2026

Edit this file to set your local paths and API keys before running any script.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
PLANTVILLAGE_DIR = ROOT / "data" / "plantvillage"
PDF_OUTPUT_DIR = ROOT / "output" / "pdf"

# Ensure output directories exist at import time
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
(ROOT / "outputs" / "reports").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs" / "models").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs" / "visualizations").mkdir(parents=True, exist_ok=True)
PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── API keys ──────────────────────────────────────────────────────────────────
# Get your free USDA key at: https://quickstats.nass.usda.gov/api
# Prefer setting USDA_API_KEY in your shell or .env instead of hardcoding it here.
USDA_API_KEY = os.getenv("USDA_API_KEY", "")

# Open-Meteo requires no API key for non-commercial use

# ── PlantVillage settings ─────────────────────────────────────────────────────
IMAGE_SIZE = 224          # MobileNetV3 input resolution
PHASH_THRESHOLD = 8       # perceptual hash deduplication threshold
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# Home-garden-relevant PlantVillage classes (prefix match)
HOME_GARDEN_CLASSES = [
    "Tomato", "Pepper", "Potato", "Strawberry",
    "Grape", "Apple", "Squash", "Corn",
]

# ImageNet normalization (for MobileNetV3 transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── USDA NASS settings ────────────────────────────────────────────────────────
USDA_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
HOME_GARDEN_CROPS = [
    "TOMATOES", "PEPPERS", "LETTUCE", "BEANS",
    "CUCUMBERS", "SQUASH", "STRAWBERRIES", "CORN",
    "GRAPES", "APPLES", "POTATOES",
]
USDA_START_YEAR = 2015
MISSING_COL_THRESHOLD = 0.30   # drop columns with >30% missing

# ── Open-Meteo settings ───────────────────────────────────────────────────────
HISTORY_DAYS    = 90
FORECAST_DAYS   = 7
WATERING_THRESHOLD_MM = 5.0     # water deficit > this → water_needed = True
FFILL_LIMIT     = 2             # max consecutive missing days to forward-fill

# Default location (Long Beach, CA) — override in your script or CLI
DEFAULT_LAT = 33.77
DEFAULT_LON = -118.19

# ── Model settings ────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70     # disease detection: below this → "possible cause"

# ── Output paths ──────────────────────────────────────────────────────────────
USDA_CLEAN_CSV     = DATA_PROCESSED / "usda_nass_clean.csv"
USDA_CLEAN_PARQUET = DATA_PROCESSED / "usda_nass_clean.parquet"
KAGGLE_CLEAN_CSV   = DATA_PROCESSED / "kaggle_agri_clean.csv"
WEATHER_CACHE_DB   = DATA_PROCESSED / "weather_cache.db"
PLANTS_KB_JSON     = DATA_PROCESSED / "plant_knowledge_base.json"
