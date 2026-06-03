"""Kumpulan fungsi dan lokasi file yang dipakai di project."""

from pathlib import Path

import joblib
import pandas as pd


# Root project
ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"

# Folder paths
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
ASSETS_DIR = ROOT_DIR / "assets"

# File paths - data
RAW_DATA_PATH = DATA_RAW_DIR / "employee_burnout_analysis.csv"
CLEANED_DATA_PATH = DATA_PROCESSED_DIR / "cleaned_dataset.csv"

# File paths - reports
CLUSTERING_RESULT_PATH = REPORTS_DIR / "clustering_result.csv"
DATA_IDENTIFICATION_REPORT_PATH = REPORTS_DIR / "data_identification_report.csv"
CLUSTER_RISK_MAPPING_PATH = REPORTS_DIR / "cluster_risk_mapping.csv"
CLUSTER_SUMMARY_PATH = REPORTS_DIR / "cluster_summary.csv"
KMEANS_METRICS_PATH = REPORTS_DIR / "kmeans_metrics.csv"
SILHOUETTE_SCORES_PATH = REPORTS_DIR / "silhouette_scores.csv"
REGRESSION_EVALUATION_PATH = REPORTS_DIR / "regression_evaluation.txt"
CLASSIFICATION_EVALUATION_PATH = REPORTS_DIR / "classification_evaluation.txt"

# File paths - models
SCALER_PATH = MODELS_DIR / "scaler.pkl"
KMEANS_PATH = MODELS_DIR / "kmeans.pkl"
REGRESSION_MODEL_PATH = MODELS_DIR / "regression_model.pkl"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "classification_model.pkl"

# File paths - assets
ELBOW_PLOT_PATH = ASSETS_DIR / "elbow_method.png"
SILHOUETTE_PLOT_PATH = ASSETS_DIR / "silhouette_score.png"
PCA_CLUSTERING_PLOT_PATH = ASSETS_DIR / "pca_clustering.png"
HEATMAP_PATH = ASSETS_DIR / "heatmap.png"
CLUSTERING_PLOT_PATH = ASSETS_DIR / "clustering.png"
DISTRIBUTION_PLOT_PATH = ASSETS_DIR / "distribution.png"

# Numeric features untuk clustering, regression, classification, dan Streamlit
NUMERIC_FEATURES = [
    "Designation",
    "Resource Allocation",
    "Mental Fatigue Score",
    "Years Working",
]


def create_directories():
    """Membuat folder yang dibutuhkan jika belum ada."""
    folders = [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        ASSETS_DIR,
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


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
    """Menyimpan model ke file .pkl."""
    create_directories()
    joblib.dump(model, path)


def load_model(path):
    """Membaca model dari file .pkl."""
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