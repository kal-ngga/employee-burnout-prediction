import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from utils import (
    CLEANED_DATA_PATH,
    CLUSTERING_RESULT_PATH,
    CLUSTER_SUMMARY_PATH,
    KMEANS_METRICS_PATH,
    SILHOUETTE_SCORES_PATH,
    ELBOW_PLOT_PATH,
    SILHOUETTE_PLOT_PATH,
    PCA_CLUSTERING_PLOT_PATH,
    NUMERIC_FEATURES,
    save_model,
    SCALER_PATH,
    KMEANS_PATH,
)


def run_clustering(cleaned_df=None):
    """Melakukan seluruh proses clustering beserta analisis ilmiah."""
    if cleaned_df is None:
        cleaned_df = pd.read_csv(CLEANED_DATA_PATH)

    # Pilih fitur numerik
    X = cleaned_df[NUMERIC_FEATURES]

    # Normalisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    save_model(scaler, SCALER_PATH)

    # 1. Elbow Method
    inertia_vals = []
    k_range = range(1, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia_vals.append(km.inertia_)
    pd.DataFrame({"K": list(k_range), "Inertia": inertia_vals}).to_csv(
        KMEANS_METRICS_PATH, index=False
    )
    plt.figure()
    plt.plot(k_range, inertia_vals, marker="o")
    plt.axvline(x=2, linestyle="--", label="Selected K = 2")
    plt.title("Elbow Method - Inertia vs. K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia (WCSS)")
    plt.legend()
    plt.savefig(ELBOW_PLOT_PATH, dpi=150)
    plt.close()

    # 2. Silhouette Score
    silhouette_vals = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_vals.append(score)
    pd.DataFrame({"K": list(k_range), "Silhouette Score": silhouette_vals}).to_csv(
        SILHOUETTE_SCORES_PATH, index=False
    )
    plt.figure()
    plt.plot(k_range, silhouette_vals, marker="o")
    plt.axvline(x=2, linestyle="--", label="Selected K = 2")
    plt.title("Silhouette Score vs. K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.savefig(SILHOUETTE_PLOT_PATH, dpi=150)
    plt.close()

    # 3. Fit K-Means dengan K=2
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    save_model(kmeans, KMEANS_PATH)

    # Tambahkan label cluster ke dataframe
    clustered_df = cleaned_df.copy()
    clustered_df["Cluster"] = cluster_labels
    clustered_df.to_csv(CLUSTERING_RESULT_PATH, index=False)

    # 4. PCA untuk visualisasi 2D
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({
        "PCA 1": pca_result[:, 0],
        "PCA 2": pca_result[:, 1],
        "Cluster": cluster_labels
    })
    plt.figure()
    sns.scatterplot(
        data=pca_df,
        x="PCA 1",
        y="PCA 2",
        hue="Cluster",
        palette=["#ff6b6b", "#2ec4b6"],
        alpha=0.8,
        s=60
    )
    plt.title("PCA Clustering Visualization (K=2)")
    plt.savefig(PCA_CLUSTERING_PLOT_PATH, dpi=150)
    plt.close()

    # ---------------------------------------------------------------
    # 5. Ringkasan cluster
    summary = (
        clustered_df.groupby("Cluster")
        .agg(
            Total_Data=("Cluster", "count"),
            Avg_Designation=("Designation", "mean"),
            Avg_Resource_Allocation=("Resource Allocation", "mean"),
            Avg_Mental_Fatigue=("Mental Fatigue Score", "mean"),
            Avg_Years_Working=("Years Working", "mean"),
            Avg_Burn_Rate=("Burn Rate", "mean"),
        )
        .reset_index()
    )
    total_records = len(clustered_df)
    summary["Percentage"] = (summary["Total_Data"] / total_records) * 100
    high_cluster = summary.loc[summary["Avg_Burn_Rate"].idxmax(), "Cluster"]
    summary["Interpretation"] = summary["Cluster"].apply(
        lambda x: "High Burnout Risk" if x == high_cluster else "Low Burnout Risk"
    )
    summary.to_csv(CLUSTER_SUMMARY_PATH, index=False)

    print("Clustering, Elbow, Silhouette, PCA dan summary selesai.")
    return clustered_df