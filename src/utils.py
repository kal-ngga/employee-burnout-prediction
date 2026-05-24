"""Kumpulan fungsi dan lokasi file yang dipakai di project."""

from pathlib import Path

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
ASSETS_DIR = ROOT_DIR / "assets"

RAW_DATA_PATH = DATA_RAW_DIR / "employee_burnout_analysis.csv"
CLEANED_DATA_PATH = DATA_PROCESSED_DIR / "cleaned_dataset.csv"
CLUSTERING_RESULT_PATH = REPORTS_DIR / "clustering_result.csv"
DATA_IDENTIFICATION_REPORT_PATH = REPORTS_DIR / "data_identification_report.csv"
CLUSTER_RISK_MAPPING_PATH = REPORTS_DIR / "cluster_risk_mapping.csv"

SCALER_PATH = MODELS_DIR / "scaler.pkl"
KMEANS_PATH = MODELS_DIR / "kmeans.pkl"
REGRESSION_MODEL_PATH = MODELS_DIR / "regression_model.pkl"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "classification_model.pkl"

NUMERIC_FEATURES = [
    "Designation",
    "Resource Allocation",
    "Mental Fatigue Score",
    "Years Working",
]


def read_raw_dataset():
    """Membaca dataset mentah."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {RAW_DATA_PATH}")

    return pd.read_csv(
        RAW_DATA_PATH,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

def save_model(model, path):
    """Menyimpan model."""
    joblib.dump(model, path)


def load_model(path):
    """Membaca model yang sudah disimpan."""
    if not path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")
    return joblib.load(path)


def classify_column_type(series):
    """Menentukan jenis data secara sederhana."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if series.name and "date" in series.name.lower():
        return "date"
    if series.dtype == "object" and series.nunique(dropna=True) > 30:
        return "string"
    return "categorical"
