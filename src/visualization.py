"""Membuat visualisasi K-Means dan analisis dataset.

Fungsi create_visualizations() akan:
1. Membuat Elbow Method (WCSS vs K) dari hasil clustering.
2. Membuat Silhouette Score vs K dari hasil clustering.
3. Membuat heatmap korelasi variabel numerik + Burn Rate.
4. Membuat distribusi Burn Rate per cluster.
5. Membuat PCA scatter plot (2D) dengan centroids.
6. Membuat scatter Mental Fatigue vs Resource Allocation.
7. Membuat overview distribusi seluruh variabel numerik.
8. Membuat overview boxplot seluruh variabel numerik.

Seluruh gambar disimpan ke folder assets.
"""

import os

from utils import ASSETS_DIR

MPL_CACHE_DIR = ASSETS_DIR / "matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import (
    CLUSTERING_RESULT_PATH,
    KMEANS_METRICS_PATH,
    SILHOUETTE_SCORES_PATH,
    ELBOW_PLOT_PATH,
    SILHOUETTE_PLOT_PATH,
    PCA_RESULT_PATH,
    PCA_CLUSTERING_PLOT_PATH,
    HEATMAP_PATH,
    DISTRIBUTION_PLOT_PATH,
    CLUSTERING_PLOT_PATH,
    NUMERIC_DISTRIBUTIONS_PATH,
    NUMERIC_BOXPLOTS_PATH,
    NUMERIC_FEATURES,
)

# pastikan seaborn punya style yang rapi
sns.set(style="whitegrid")


def create_numeric_overview(clustered_df):
    """Membuat histogram dan boxplot untuk seluruh variabel numerik."""
    numeric_columns = [
        column
        for column in NUMERIC_FEATURES + ["Burn Rate"]
        if clustered_df[column].nunique() > 1
    ]
    row_count = (len(numeric_columns) + 1) // 2

    figure, axes = plt.subplots(row_count, 2, figsize=(12, 4 * row_count), squeeze=False)
    figure.suptitle("Distribusi Variabel Numerik", fontsize=16, fontweight="bold")
    for axis, column in zip(axes.flat, numeric_columns):
        sns.histplot(
            data=clustered_df,
            x=column,
            bins=20,
            color="#72b7d2",
            edgecolor="black",
            ax=axis,
        )
        axis.set_title(column, fontweight="bold")
        axis.set_ylabel("Jumlah Data")
    for axis in axes.flat[len(numeric_columns):]:
        axis.remove()
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(NUMERIC_DISTRIBUTIONS_PATH, dpi=300)
    plt.close(figure)

    figure, axes = plt.subplots(row_count, 2, figsize=(12, 4 * row_count), squeeze=False)
    figure.suptitle("Boxplot Variabel Numerik", fontsize=16, fontweight="bold")
    for axis, column in zip(axes.flat, numeric_columns):
        sns.boxplot(data=clustered_df, y=column, color="#df8989", ax=axis)
        axis.set_title(column, fontweight="bold")
        axis.set_xlabel("")
    for axis in axes.flat[len(numeric_columns):]:
        axis.remove()
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(NUMERIC_BOXPLOTS_PATH, dpi=300)
    plt.close(figure)


def create_visualizations(clustered_df=None):
    """Membuat seluruh visualisasi dan menyimpannya ke folder assets."""
    # Load hasil clustering jika belum diberikan
    if clustered_df is None:
        clustered_df = pd.read_csv(CLUSTERING_RESULT_PATH)
    if "Cluster" not in clustered_df.columns:
        raise ValueError(
            "Kolom 'Cluster' tidak ditemukan. Jalankan run_clustering() terlebih dahulu."
        )

    # ----------------------------------------------------------------------
    # 1) Elbow Method
    kmeans_metrics = pd.read_csv(KMEANS_METRICS_PATH)
    plt.figure(figsize=(8, 5))
    plt.plot(kmeans_metrics["K"], kmeans_metrics["Inertia"], marker="o")
    plt.axvline(x=2, linestyle="--", color="grey", label="K=2")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia (WCSS)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ELBOW_PLOT_PATH, dpi=300)
    plt.close()

    # ----------------------------------------------------------------------
    # 2) Silhouette Score
    silhouette_scores = pd.read_csv(SILHOUETTE_SCORES_PATH)
    plt.figure(figsize=(8, 5))
    plt.plot(
        silhouette_scores["K"],
        silhouette_scores["Silhouette Score"],
        marker="o",
    )
    plt.axvline(x=2, linestyle="--", color="grey", label="K=2")
    plt.title("Silhouette Score vs K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SILHOUETTE_PLOT_PATH, dpi=300)
    plt.close()

    # ----------------------------------------------------------------------
    # 3) Heatmap Korelasi
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        clustered_df[NUMERIC_FEATURES + ["Burn Rate"]].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=300)
    plt.close()

    # ----------------------------------------------------------------------
    # 4) Distribusi Burn Rate per cluster
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=clustered_df,
        x="Burn Rate",
        hue="Cluster",
        kde=True,
        bins=30,
        alpha=0.6,
        palette=["#2ec4b6", "#ff8a65"],
    )
    plt.title("Burn Rate Distribution by Cluster")
    plt.xlabel("Burn Rate")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(DISTRIBUTION_PLOT_PATH, dpi=300)
    plt.close()

    # ----------------------------------------------------------------------
    # 5) PCA Visualization dengan centroid
    pca_df = pd.read_csv(PCA_RESULT_PATH)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="PCA 1",
        y="PCA 2",
        hue="Cluster",
        palette=["#ff6b6b", "#2ec4b6"],
        alpha=0.7,
        s=60,
    )
    pca_centroids = pca_df.groupby("Cluster")[["PCA 1", "PCA 2"]].mean()
    plt.scatter(
        pca_centroids["PCA 1"],
        pca_centroids["PCA 2"],
        marker="X",
        s=200,
        c="yellow",
        edgecolor="black",
    )
    plt.title("PCA Clustering Visualization (K=2)")
    plt.tight_layout()
    plt.savefig(PCA_CLUSTERING_PLOT_PATH, dpi=300)
    plt.close()

    # ----------------------------------------------------------------------
    # 6) Scatter Mental Fatigue vs Resource Allocation
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=clustered_df,
        x="Mental Fatigue Score",
        y="Resource Allocation",
        hue="Cluster",
        palette=["#ff6b6b", "#2ec4b6"],
        alpha=0.7,
        s=50,
    )
    plt.title("Mental Fatigue vs Resource Allocation by Cluster")
    plt.xlabel("Mental Fatigue Score")
    plt.ylabel("Resource Allocation")
    plt.tight_layout()
    plt.savefig(CLUSTERING_PLOT_PATH, dpi=300)
    plt.close()

    # ----------------------------------------------------------------------
    # 7-8) Overview distribusi dan boxplot variabel numerik
    create_numeric_overview(clustered_df)

    print("Semua visualisasi telah dibuat dan disimpan ke folder assets.")
