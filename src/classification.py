"""Model klasifikasi untuk memprediksi cluster."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from utils import (
    CLASSIFICATION_MODEL_PATH,
    CLUSTERING_RESULT_PATH,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    save_model,
)


def run_classification(clustered_df=None):
    """Melatih model klasifikasi dan menyimpan hasil evaluasi."""
    if clustered_df is None:
        clustered_df = pd.read_csv(CLUSTERING_RESULT_PATH)

    x = clustered_df[NUMERIC_FEATURES]
    y = clustered_df["Cluster"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    evaluation = [
        "Classification Evaluation - RandomForestClassifier",
        "Target variable: Cluster",
        f"Accuracy: {accuracy_score(y_test, predictions):.4f}",
        "",
        "Confusion Matrix:",
        str(confusion_matrix(y_test, predictions)),
        "",
        "Classification Report:",
        classification_report(y_test, predictions),
    ]

    evaluation_path = REPORTS_DIR / "classification_evaluation.txt"
    evaluation_path.write_text("\n".join(evaluation), encoding="utf-8")
    save_model(model, CLASSIFICATION_MODEL_PATH)

    print("Classification: RandomForestClassifier trained with Cluster as target.")
    print(f"Classification evaluation saved to {evaluation_path}")

    return model

if __name__ == "__main__":
    run_classification()
