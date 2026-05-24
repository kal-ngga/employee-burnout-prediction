"""Membuat cluster dengan K-Means."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils import (
    CLEANED_DATA_PATH,
    CLUSTERING_RESULT_PATH,
    CLUSTER_RISK_MAPPING_PATH,
    KMEANS_PATH,
    NUMERIC_FEATURES,
    SCALER_PATH,
    save_model,
)


def run_clustering(cleaned_df=None):
    """Melatih K-Means dan menyimpan hasil clustering."""
    if cleaned_df is None:
        cleaned_df = pd.read_csv(CLEANED_DATA_PATH)

    x = cleaned_df[NUMERIC_FEATURES]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clustered_df = cleaned_df.copy()
    clustered_df["Cluster"] = kmeans.fit_predict(x_scaled)

    risk_mapping = (
        clustered_df.groupby("Cluster")[["Burn Rate", "Mental Fatigue Score"]]
        .mean()
        .reset_index()
        .sort_values(["Burn Rate", "Mental Fatigue Score"])
    )
    risk_mapping["Risk Label"] = ["Low Burnout Risk", "High Burnout Risk"]
    risk_mapping["Recommendation"] = [
        "Maintain current workload balance and keep monitoring fatigue levels.",
        "Reduce workload pressure, improve rest time, and review resource allocation.",
    ]

    clustered_df.to_csv(CLUSTERING_RESULT_PATH, index=False)
    risk_mapping.to_csv(CLUSTER_RISK_MAPPING_PATH, index=False)
    save_model(scaler, SCALER_PATH)
    save_model(kmeans, KMEANS_PATH)

    print("Data Clustering: K-Means completed with 2 clusters.")
    print(f"Clustering result saved to {CLUSTERING_RESULT_PATH}")
    print(f"Cluster risk mapping saved to {CLUSTER_RISK_MAPPING_PATH}")

    return clustered_df


if __name__ == "__main__":
    run_clustering()
