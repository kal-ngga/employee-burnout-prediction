"""Main entry point for running the full CRISP-DM project pipeline."""

import os

# Silence CPU-count warnings from joblib in restricted local environments.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

from data_preparation import prepare_data
from clustering import run_clustering
from regression import run_regression
from classification import run_classification
from visualization import create_visualizations


def main() -> None:
    """Run all project stages from data preparation to model evaluation."""
    print("Starting Employee Burnout Prediction pipeline...")
    cleaned_df = prepare_data()
    clustered_df = run_clustering(cleaned_df)
    run_regression(clustered_df)
    run_classification(clustered_df)
    create_visualizations(clustered_df)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
