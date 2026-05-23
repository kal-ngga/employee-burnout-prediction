"""Visualization step for correlation, clustering, and distribution charts."""

import os

from utils import ASSETS_DIR

# Keep Matplotlib cache inside the project so the pipeline works in restricted shells.
MPL_CACHE_DIR = ASSETS_DIR / "matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import CLUSTERING_RESULT_PATH, NUMERIC_FEATURES


def create_visualizations(clustered_df: pd.DataFrame | None = None) -> None:
    """Create and save the visual assets required by the project."""
    if clustered_df is None:
        clustered_df = pd.read_csv(CLUSTERING_RESULT_PATH)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.heatmap(clustered_df[NUMERIC_FEATURES + ["Burn Rate", "Cluster"]].corr(), annot=True, cmap="viridis")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=clustered_df,
        x="Resource Allocation",
        y="Mental Fatigue Score",
        hue="Cluster",
        palette="Set2",
    )
    plt.title("K-Means Clustering Result")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "clustering.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.histplot(data=clustered_df, x="Burn Rate", hue="Cluster", kde=True, palette="Set2")
    plt.title("Burn Rate Distribution by Cluster")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "distribution.png", dpi=150)
    plt.close()

    print(f"Visualizations saved to {ASSETS_DIR}")


if __name__ == "__main__":
    create_visualizations()
