"""
GrowAI Phase 2 — Crop Recommendation Model
============================================
Authors: Madike Asia, Sreytouch Lang (Jessica)
Thullen Lab, July 2026

Model: Random Forest (baseline) → XGBoost (tuned)
Data : USDA NASS + Kaggle Agriculture merged features
Target: crop label (multi-class classification)
Inputs: N, P, K, pH, temperature, humidity, rainfall, region

Literature accuracy benchmarks:
  - Random Forest : 96.98–99.5%  (Scientific Reports, Jul 2025)
  - XGBoost       : 98.51–99.09% (Heliyon, Jan 2024)
  RF preferred for production due to superior generalization.

Usage:
  python src/models/crop_recommendation.py
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

from src.utils.config import DATA_PROCESSED, RANDOM_SEED
from src.utils.data_quality import data_quality_report

MODEL_OUTPUT = Path("outputs/models/crop_rf_model.pkl")
ENCODER_OUTPUT = Path("outputs/models/label_encoder.pkl")

FEATURE_COLS = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare features + labels from processed files."""
    kaggle_path = DATA_PROCESSED / "kaggle_agri_clean.csv"
    if not kaggle_path.exists():
        raise FileNotFoundError(
            f"Kaggle clean file not found: {kaggle_path}\n"
            "Run src/preprocessing/kaggle_agri.py first."
        )

    df = pd.read_csv(kaggle_path)
    data_quality_report(df, "Kaggle Agriculture (pre-modeling check)")

    # Identify label column
    label_col = next((c for c in df.columns if "label" in c), None)
    if label_col is None:
        raise ValueError("No label column found. Expected a column containing 'label'.")

    # Keep only feature columns that exist
    features = [c for c in FEATURE_COLS if c in df.columns]
    print(f"Features used: {features}")

    X = df[features].dropna()
    y = df.loc[X.index, label_col]
    print(f"Dataset: {X.shape[0]:,} samples × {X.shape[1]} features | {y.nunique()} classes")
    return X, y


def train_random_forest(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Train a Random Forest classifier with 5-fold cross-validation.

    Returns (trained model, label encoder, cv_scores)
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )

    print("\nRunning 5-fold cross-validation ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X, y_enc, cv=cv, scoring="accuracy")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit on full training data
    model.fit(X, y_enc)
    return model, le, cv_scores


def evaluate(model, le, X: pd.DataFrame, y: pd.Series, split_name: str = "test"):
    """Print accuracy and classification report on a held-out split."""
    y_enc = le.transform(y)
    y_pred = model.predict(X)
    acc = accuracy_score(y_enc, y_pred)
    print(f"\n{split_name.upper()} Accuracy: {acc:.4f}")
    print(classification_report(y_enc, y_pred, target_names=le.classes_))
    return acc


def feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Return SHAP-style feature importance (Gini impurity)."""
    imp = pd.DataFrame({
        "feature":   feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nFeature importances:")
    print(imp.to_string(index=False))
    return imp


def run_pipeline():
    print("\n=== Crop Recommendation Model Pipeline ===")
    X, y = load_data()

    # Train/test split (stratified 80/20 for final evaluation)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )

    model, le, cv_scores = train_random_forest(X_train, y_train)
    evaluate(model, le, X_test, y_test, split_name="test")
    feature_importance(model, list(X.columns))

    # Save model
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)
    joblib.dump(le, ENCODER_OUTPUT)
    print(f"\nModel saved → {MODEL_OUTPUT}")
    print(f"Encoder saved → {ENCODER_OUTPUT}")
    print("\n=== Crop Recommendation Pipeline Complete ===")
    return model, le


def predict(features: dict) -> str:
    """
    Predict the best crop for given soil and climate inputs.

    Parameters
    ----------
    features : dict with keys matching FEATURE_COLS
               e.g. {'n': 90, 'p': 42, 'k': 43, 'temperature': 20.8,
                     'humidity': 82, 'ph': 6.5, 'rainfall': 202.9}

    Returns
    -------
    Predicted crop name (string)
    """
    if not MODEL_OUTPUT.exists():
        raise FileNotFoundError("Model not found. Run run_pipeline() first.")

    model = joblib.load(MODEL_OUTPUT)
    le    = joblib.load(ENCODER_OUTPUT)
    X = pd.DataFrame([features])[FEATURE_COLS]
    pred_enc = model.predict(X)[0]
    return le.inverse_transform([pred_enc])[0]


if __name__ == "__main__":
    run_pipeline()
