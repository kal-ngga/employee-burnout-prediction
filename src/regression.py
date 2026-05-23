"""Regression step using RandomForestRegressor to predict the Cluster target."""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils import CLUSTERING_RESULT_PATH, NUMERIC_FEATURES, REGRESSION_MODEL_PATH, REPORTS_DIR, save_model


def run_regression(clustered_df: pd.DataFrame | None = None) -> RandomForestRegressor:
    """Train and evaluate a RandomForestRegressor using Cluster as target."""
    if clustered_df is None:
        clustered_df = pd.read_csv(CLUSTERING_RESULT_PATH)

    x = clustered_df[NUMERIC_FEATURES]
    y = clustered_df["Cluster"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    evaluation = [
        "Regression Evaluation - RandomForestRegressor",
        f"Target variable: Cluster",
        f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.4f}",
        f"Mean Squared Error: {mean_squared_error(y_test, predictions):.4f}",
        f"R2 Score: {r2_score(y_test, predictions):.4f}",
    ]

    evaluation_path = REPORTS_DIR / "regression_evaluation.txt"
    evaluation_path.write_text("\n".join(evaluation), encoding="utf-8")
    save_model(model, REGRESSION_MODEL_PATH)

    print("Regression: RandomForestRegressor trained with Cluster as target.")
    print(f"Regression evaluation saved to {evaluation_path}")

    return model


if __name__ == "__main__":
    run_regression()
