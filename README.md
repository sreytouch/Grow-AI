# GrowAI — Phase 2: Dataset Pre-Processing & Data Modeling

**AI for Sustainable Home Gardening and Small Farms**

> **Authors:** Madike Asia · Sreytouch Lang (Jessica)  
> **Supervisor:** Prof. Eve Thullen — Thullen Lab  
> **Date:** July 2026  
> **Phase:** 2 — Dataset Pre-Processing, Baseline Modeling, Sustainability Goal Hierarchy

---

## Project Overview

GrowAI is an AI-powered platform that helps beginner home gardeners and small farm operators with:

- **Crop recommendation** — what to grow based on soil, weather, and location
- **Plant disease detection** — photo-based diagnosis to reduce pesticide use
- **Smart watering** — ET₀-driven irrigation scheduling
- **Chatbot guidance** — plain-language explanations powered by RAG

Phase 2 covers:
1. Pre-processing all five confirmed datasets
2. Training baseline AI models for each module
3. Ranking the four sustainability goals based on model results

---

## Repository Structure

```
GrowAI/
│
├── README.md                        ← You are here
├── requirements.txt                 ← Python dependencies
├── .gitignore
│
├── notebooks/
│   └── GrowAI_Dataset_Preprocessing.ipynb   ← Main pre-processing notebook
│
├── src/
│   ├── preprocessing/
│   │   ├── plantvillage.py          ← Image pipeline (resize, augment, split)
│   │   ├── usda_nass.py             ← USDA API fetch + tabular cleaning
│   │   ├── open_meteo.py            ← Weather API + ET₀ watering engine
│   │   ├── kaggle_agri.py           ← Kaggle CSV cleaning + feature engineering
│   │   └── indoor_plants.py         ← Knowledge base builder (JSON export)
│   │
│   ├── models/
│   │   ├── crop_recommendation.py   ← Random Forest / XGBoost crop model
│   │   ├── watering_engine.py       ← FAO-56 ET₀ rule engine
│   │
│   └── utils/
│       ├── data_quality.py          ← data_quality_report() function
│       ├── config.py                ← Shared paths, API keys, constants
│       └── export_dataset_pdfs.py   ← PDF export helper
│
├── data/
│   ├── raw/                         ← Downloaded datasets go here (not in Git)
│   │   ├── Crop_recommendation.csv
│   │   └── indoor_plants.csv
│   ├── plantvillage/                ← PlantVillage images (not in Git)
│   └── processed/                   ← Cleaned outputs (auto-created by scripts)
│
├── output/
│   └── pdf/                         ← Generated dataset PDFs
│
├── outputs/
│   ├── reports/                     ← Runtime reports (created as needed)
│   ├── models/                      ← Saved model weights (.pkl, .pt)
│   └── visualizations/              ← Plots and charts
│
└── docs/
    ├── DATASET_GUIDE.md             ← How to download each dataset
    └── MODEL_GUIDE.md               ← Model architecture and training details
```

Phase 3 modules such as disease detection and chatbot RAG are documented in [`docs/MODEL_GUIDE.md`](docs/MODEL_GUIDE.md) but are not implemented in this Phase 2 repo yet.

---

## Quick Start

### 1. Clone / download the project

```bash
git clone git@github.com:SreytouchLang/GrowAI.git
cd GrowAI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download datasets

See [`docs/DATASET_GUIDE.md`](docs/DATASET_GUIDE.md) for step-by-step download instructions for each dataset.

### 4. Set your API key and paths

Set your USDA key in your shell before running the USDA pipeline:

```bash
export USDA_API_KEY="your_api_key_here"
```

You can still edit `src/utils/config.py` for local path defaults:

```python
PLANTVILLAGE_DIR = "data/plantvillage"
```

### 5. Run the notebook

```bash
jupyter notebook notebooks/GrowAI_Dataset_Preprocessing.ipynb
jupyter notebook notebooks/GrowAI_Crop_Recommendation_Model.ipynb
```

Run cells from top to bottom. Each section is self-contained.

### 6. Or run individual scripts

```bash
# Pre-process USDA NASS data
python src/preprocessing/usda_nass.py

# Pre-process PlantVillage images
python src/preprocessing/plantvillage.py

# Fetch and process weather data (Long Beach, CA example)
python src/preprocessing/open_meteo.py --lat 33.77 --lon -118.19
```

---

## Sustainability Goals (to be ranked after Phase 2)

| Priority | Goal | Dataset | Model |
|----------|------|---------|-------|
| 1 (TBD) | CO₂ Emission Reduction | USDA NASS + Open-Meteo | Crop recommendation |
| 2 (TBD) | Water Savings | Open-Meteo | ET₀ watering engine |
| 3 (TBD) | Cost Efficiency | Kaggle + USDA NASS | Yield/cost scoring |
| 4 (TBD) | Low Pesticide Use | PlantVillage | Disease detection |

Rankings finalized after baseline model evaluation.

---

## Literature

All pre-processing decisions are grounded in peer-reviewed work (2024–2026). See the full reference list in [`docs/MODEL_GUIDE.md`](docs/MODEL_GUIDE.md) and the Phase 2 research paper (`GrowAI_PreProcessing_Paper.docx`).

---

## License

Research use only. Not for commercial deployment without license review (Open-Meteo CC BY 4.0 non-commercial applies to weather data).
