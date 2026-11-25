# GrowAI — Model Architecture Guide

**Authors:** Madike Asia · Sreytouch Lang (Jessica)  
**Thullen Lab, July 2026**

---

## Overview

GrowAI uses a **hybrid model architecture** — four specialized components, not a single monolithic AI. Each component is the right tool for its specific task.

| Component | Model | Task | Target Metric |
|-----------|-------|------|--------------|
| Crop Recommendation | Random Forest / XGBoost | Multi-class classification | Accuracy >96%, F1 per class |
| Disease Detection | MobileNetV3 (transfer learning) | Image classification (38 classes) | Val accuracy >85% on home-garden classes |
| Watering Engine | FAO-56 ET₀ rule-based | Daily irrigation scheduling | Water deficit vs. manual reference |
| Chatbot | RAG over Indoor Plants JSON | Natural language Q&A | Factual response rate (human eval) |

---

## Module 1 — Crop Recommendation

**File:** `src/models/crop_recommendation.py`

### Why Random Forest?
Multiple peer-reviewed studies benchmark Random Forest at 96.98–99.5% accuracy on NPK-based crop datasets (Scientific Reports, Jul 2025; Heliyon, Jan 2024). XGBoost achieves slightly higher accuracy (98.51–99.09%) but has lower generalization when data distribution shifts — expected for regional crop data. Both are evaluated; Random Forest is preferred for production.

### Input Features
| Feature | Unit | Description |
|---------|------|-------------|
| N | mg/kg | Nitrogen content of soil |
| P | mg/kg | Phosphorus content of soil |
| K | mg/kg | Potassium content of soil |
| temperature | °C | Average air temperature |
| humidity | % | Relative humidity |
| ph | — | Soil pH (0–14) |
| rainfall | mm | Annual/seasonal rainfall |

### Training Protocol
- **Split:** 80% train / 20% test (stratified by class)
- **CV:** 5-fold StratifiedKFold on training set
- **Hyperparameters:** n_estimators=200, class_weight='balanced'
- **Interpretability:** SHAP / Gini feature importance reported

---

## Module 2 — Plant Disease Detection

**File:** `src/models/disease_detection.py` *(Phase 3 — full training loop)*

### Why MobileNetV3?
MobileNetV3 offers competitive accuracy (~97–99% on PlantVillage in-distribution) with a 4× smaller footprint than ResNet50, making it suitable for mobile deployment. EfficientNet-B0 and MobileNetV2 are evaluated as alternatives (MDPI Sensors, Jan 2025; Springer IoT, Feb 2026).

### Transfer Learning Strategy
1. Load MobileNetV3 pre-trained on ImageNet
2. Replace final classification layer with a new softmax head sized to the number of retained home-garden classes
3. Freeze all layers except the last 2 blocks for first 5 epochs
4. Unfreeze all layers and fine-tune at a lower learning rate (1e-5)

### Confidence Threshold
Predictions below **70% confidence** are reported as `"possible cause"` rather than a definitive diagnosis. This prevents inappropriate pesticide recommendations when the model is uncertain.

### Dataset
PlantVillage filtered to home-garden classes (~14 of 38 original classes), augmented, 70/15/15 stratified split. See `src/preprocessing/plantvillage.py`.

---

## Module 3 — Watering Rule Engine

**File:** `src/models/watering_engine.py`

### Why rule-based, not ML?
- Transparent and explainable — users and Prof. Thullen can audit the logic
- No overfitting risk from a live API with variable data quality
- FAO-56 Penman-Monteith ET₀ is the globally accepted standard (FAO, 1998)
- Phase 3 may introduce a learned model if rule-based accuracy proves insufficient

### Formula
```
water_deficit_mm = max(ET₀ − daily_precipitation, 0)
water_needed     = water_deficit_mm > crop_threshold
```

### Crop Thresholds (mm/day)
| Crop | Threshold |
|------|-----------|
| Tomato | 4.0 |
| Lettuce | 3.0 |
| Pepper | 4.5 |
| Strawberry | 3.5 |
| Cucumber | 5.0 |
| Basil/herbs | 3.0 |
| Default | 5.0 |

---

## Module 4 — LLM Chatbot (RAG)

**File:** `src/models/chatbot_rag.py` *(Phase 3)*

### Architecture
Retrieval-Augmented Generation (RAG):
1. User query + current model outputs sent to retrieval layer
2. Relevant plant-care records retrieved from the Indoor Plants JSON knowledge base
3. LLM generates an explanation **grounded in the retrieved records** — not from free-form memory
4. RAG grounding prevents hallucination of agronomic advice

### Why RAG over fine-tuning?
- Knowledge base is small (~200–500 records) — fine-tuning would overfit
- RAG allows the knowledge base to be updated without retraining
- Grounded responses are more defensible for plant health advice

---

## References

- Howard, A. G. et al. (2017). MobileNets. arXiv:1704.04861.
- Nature Scientific Reports (Jul 2025). Advancing crop recommendation with XGBoost. doi:10.1038/s41598-025-07003-8
- Heliyon (Jan 2024). ML crop recommendation under NPK and climate variables. doi:10.1016/j.heliyon.2024.e25112
- MDPI Sensors (Jan 2025). Plant Leaf Disease Detection: A Multi-Dataset Approach. doi:10.3390/s8010004
- FAO (1998). Crop Evapotranspiration. Irrigation & Drainage Paper 56.
- Frontiers Sustainable Food Systems (Feb 2026). Accurate ET₀ estimation. doi:10.3389/fsufs.2026.1734366
- Mohanty et al. (2016). Using deep learning for plant disease detection. Frontiers in Plant Science.
