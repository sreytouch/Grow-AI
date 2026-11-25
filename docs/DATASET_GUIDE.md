# GrowAI — Dataset Download Guide

**Authors:** Madike Asia · Sreytouch Lang (Jessica)  
**Thullen Lab, July 2026**

This guide explains how to download and place each dataset before running the pre-processing pipelines.

---

## 1. PlantVillage (Image dataset)

**Size:** ~1.3 GB (54,305 images, 38 classes)  
**License:** CC BY-SA 4.0 — free for research and commercial use

### Steps
1. Go to: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Click **Download** (requires a free Kaggle account)
3. Unzip the downloaded file
4. Move the class folders into `data/plantvillage/`

```
data/plantvillage/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
└── ...
```

### Alternative — TensorFlow Datasets
```python
import tensorflow_datasets as tfds
ds = tfds.load('plant_village', split='train', shuffle_files=True)
```

---

## 2. USDA NASS Quick Stats (API)

**Size:** Dynamic (API pull)  
**License:** Public Domain — free for all uses

### Steps
1. Register for a free API key: https://quickstats.nass.usda.gov/api
2. You'll receive an email with your key (usually instant)
3. Set it in your shell before running the pipeline:
   ```bash
   export USDA_API_KEY="your_key_here"
   ```
4. Or, if you prefer, edit `src/utils/config.py` and set the fallback value there.
5. Run the pipeline — data is fetched automatically:
   ```bash
   python src/preprocessing/usda_nass.py
   ```

No file download needed.

---

## 3. Open-Meteo (Weather API)

**Size:** Dynamic (API pull)  
**License:** CC BY 4.0 (non-commercial use) — review before commercial deployment

### Steps
No API key required. Just run:
```bash
python src/preprocessing/open_meteo.py --lat 33.77 --lon -118.19
```

Replace `33.77 / -118.19` with your actual location coordinates.

**To find your coordinates:**
- Google Maps → right-click on your location → "What's here?" → shows lat/lon
- Or use https://www.latlong.net/

---

## 4. Kaggle Agriculture & Farming

**Recommended dataset:** Crop Recommendation Dataset by Atharva Ingle  
**Size:** ~150 KB (2,200 rows)  
**License:** CC0 (Public Domain)

### Steps

**Option A — Kaggle CLI (recommended)**
```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/kaggle.json
kaggle datasets download -d atharvaingle/crop-recommendation-dataset
unzip crop-recommendation-dataset.zip -d data/raw/
```

**Option B — Manual download**
1. Go to: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. Click **Download**
3. Place `Crop_recommendation.csv` in `data/raw/`

```
data/raw/
└── Crop_recommendation.csv
```

> **License note:** This dataset is CC0 (public domain). Other Kaggle agriculture datasets may have different licenses — always check the dataset card before using in the pipeline.

---

## 5. Indoor House Plants Dataset

**Size:** ~50 KB  
**License:** Public / CC

### Steps
1. Search Kaggle for: `indoor house plants care instructions`
2. Recommended: https://www.kaggle.com/datasets/russellpecar/indoor-house-plants-with-care-instructions
3. Download and place in `data/raw/indoor_plants.csv`

```
data/raw/
└── indoor_plants.csv
```

---

## Final folder check

After downloading everything, your structure should look like:

```
data/
├── raw/
│   ├── Crop_recommendation.csv    ✓
│   └── indoor_plants.csv          ✓
├── plantvillage/
│   ├── Apple___Apple_scab/        ✓
│   ├── Tomato___Early_blight/     ✓
│   └── ...                        ✓
└── processed/                     ← auto-created when you run pipelines
```

Once this is in place, you're ready to run the notebook or individual scripts.
