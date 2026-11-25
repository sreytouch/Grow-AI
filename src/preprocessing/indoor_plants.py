"""
GrowAI Phase 2 — Indoor House Plants Knowledge Base Builder
============================================================
Authors: Madike Asia, Sreytouch Lang (Jessica)
Thullen Lab, July 2026

This script converts the Indoor House Plants CSV dataset into a structured
JSON knowledge base for the GrowAI LLM chatbot RAG (Retrieval-Augmented
Generation) pipeline.

This is NOT a model training dataset — it is a retrieval index for the chatbot.

Pipeline:
  1. Load and standardize plant names (lowercase, clean special chars)
  2. Map watering_frequency text → numeric days interval
  3. Extract structured fields: sunlight, soil, humidity
  4. Tag outdoor-applicable plants (basil, tomato, lettuce, etc.)
  5. Export as JSON array for chatbot ingestion

Download:
  Search Kaggle for "indoor house plants dataset care instructions"
"""

import json
import re
import pandas as pd
from pathlib import Path

from src.utils.config import DATA_RAW, DATA_PROCESSED, PLANTS_KB_JSON

# Watering frequency text → numeric days mapping
WATERING_MAP = {
    "daily":            1,
    "every day":        1,
    "once a day":       1,
    "twice a week":     3,
    "every 3 days":     3,
    "every three days": 3,
    "every 4 days":     4,
    "every 5 days":     5,
    "every 6 days":     6,
    "weekly":           7,
    "once a week":      7,
    "every week":       7,
    "every 10 days":    10,
    "every 2 weeks":    14,
    "biweekly":         14,
    "twice a month":    14,
    "every 3 weeks":    21,
    "monthly":          30,
    "every month":      30,
    "once a month":     30,
    "every 6 weeks":    42,
    "rarely":           60,
}

PHRASE_WATERING_HINTS = [
    ("must not be dry", 2),
    ("keep moist", 3),
    ("slightly moist", 4),
    ("half dry", 7),
    ("can be a bit dry", 7),
    ("allow the soil to dry", 10),
    ("must be dry between watering", 10),
    ("soil is dry", 10),
]

# Plants that can grow outdoors in home gardens
OUTDOOR_PLANTS = {
    "tomato", "pepper", "basil", "mint", "rosemary", "lavender",
    "thyme", "sage", "parsley", "chives", "dill", "oregano",
    "lettuce", "spinach", "kale", "arugula", "chard",
    "strawberry", "blueberry", "raspberry",
    "cucumber", "squash", "zucchini", "pumpkin",
    "bean", "pea", "carrot", "radish", "beet",
    "sunflower", "marigold", "nasturtium",
    "aloe", "succulent",
}


def parse_watering_days(value: str) -> int | None:
    """Convert free-text watering frequency to numeric days."""
    if pd.isna(value):
        return None
    val = str(value).lower().strip()

    for key, days in WATERING_MAP.items():
        if key in val:
            return days

    match = re.search(r"every\s+(\d+)\s+(day|week|month)", val)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        if unit == "day":
            return value
        if unit == "week":
            return value * 7
        if unit == "month":
            return value * 30

    for phrase, days in PHRASE_WATERING_HINTS:
        if phrase in val:
            return days

    return None


def is_outdoor_applicable(name: str) -> bool:
    """Check if a plant can grow outdoors in a home garden."""
    name_lower = str(name).lower()
    return any(outdoor in name_lower for outdoor in OUTDOOR_PLANTS)


def preprocess_plants_knowledge(filepath: str | Path = None) -> pd.DataFrame:
    """
    Build a structured plant knowledge base from the Indoor Plants CSV.

    Parameters
    ----------
    filepath : path to the CSV (defaults to data/raw/indoor_plants.csv)

    Returns
    -------
    Cleaned DataFrame; also exports JSON to data/processed/plant_knowledge_base.json
    """
    if filepath is None:
        filepath = DATA_RAW / "indoor_plants.csv"

    print("\n=== Indoor Plants Knowledge Base Pipeline ===")
    df = pd.read_csv(filepath)
    print(f"Raw shape: {df.shape}")
    print(f"Columns  : {list(df.columns)}")

    # 1. Standardize column names
    df.columns = [
        c.lower().strip()
         .replace(" ", "_")
         .replace("/", "_")
         .replace("(", "")
         .replace(")", "")
        for c in df.columns
    ]

    # 2. Standardize plant name
    name_col = next((c for c in df.columns if "name" in c or "plant" in c), None)
    if name_col:
        df["plant_name_clean"] = (
            df[name_col].astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"[^a-z0-9\s\-]", "", regex=True)
        )

    # 3. Map watering frequency → numeric days
    water_col = next((c for c in df.columns if "water" in c), None)
    if water_col:
        df["watering_days"] = df[water_col].apply(parse_watering_days)
        mapped = df["watering_days"].notna().sum()
        print(f"  Watering frequency mapped: {mapped}/{len(df)} rows")

    # 4. Tag outdoor-applicable plants
    if name_col:
        df["applicable_to_outdoor"] = df[name_col].apply(is_outdoor_applicable)
        outdoor_count = df["applicable_to_outdoor"].sum()
        print(f"  Outdoor-applicable plants: {outdoor_count}")

    # 5. Export as JSON knowledge base
    records = df.to_dict(orient="records")

    # Clean non-serializable types
    def clean_val(v):
        if pd.isna(v) if not isinstance(v, (list, dict)) else False:
            return None
        if isinstance(v, (int, float)):
            return None if pd.isna(v) else v
        return str(v)

    clean_records = [
        {k: clean_val(v) for k, v in rec.items()}
        for rec in records
    ]

    PLANTS_KB_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PLANTS_KB_JSON, "w", encoding="utf-8") as f:
        json.dump(clean_records, f, indent=2, default=str)

    print(f"Knowledge base saved → {PLANTS_KB_JSON} ({len(clean_records)} entries)")
    print("\n=== Indoor Plants Pipeline Complete ===")
    return df


def load_knowledge_base(path: Path = PLANTS_KB_JSON) -> list[dict]:
    """Load the JSON knowledge base for chatbot retrieval."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    df = preprocess_plants_knowledge()
    if not df.empty:
        print(df[["plant_name_clean", "watering_days", "applicable_to_outdoor"]].head(10))
