import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from utils import (
    CLUSTERING_RESULT_PATH,
    CLASSIFICATION_MODEL_PATH,
    CLASSIFICATION_EVALUATION_PATH,
    REPORTS_DIR,
    NUMERIC_FEATURES,
    create_directories
)

def run_classification():
    create_directories()
    df = pd.read_csv(CLUSTERING_RESULT_PATH)

    X = df[NUMERIC_FEATURES]
    y = df["Cluster"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)

    # 2. Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)

    # Cross‑validation (accuracy) cv=5
    rf_cv_scores = cross_val_score(rf_clf, X, y, cv=5, scoring="accuracy")
    log_cv_scores = cross_val_score(log_reg, X, y, cv=5, scoring="accuracy")

    # Evaluasi pada test set
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    log_accuracy = accuracy_score(y_test, y_pred_log)

    # Confusion matrix dan classification report untuk model Random Forest
    rf_cm = confusion_matrix(y_test, y_pred_rf)
    rf_report = classification_report(y_test, y_pred_rf)

    # Simpan model Random Forest sebagai model utama
    joblib.dump(rf_clf, CLASSIFICATION_MODEL_PATH)

    with open(CLASSIFICATION_EVALUATION_PATH, "w") as file:
        file.write("CLASSIFICATION EVALUATION\n")
        file.write("=========================\n\n")
        file.write("Model 1: RandomForestClassifier\n")
        file.write(f"Test Accuracy           : {rf_accuracy:.4f}\n")
        file.write(f"Cross-Validation Accuracy (mean) : {np.mean(rf_cv_scores):.4f}\n\n")
        file.write("Classification Report:\n")
        file.write(f"{rf_report}\n")
        file.write("Confusion Matrix:\n")
        file.write(f"{rf_cm}\n\n")

        file.write("Model 2: LogisticRegression\n")
        file.write(f"Test Accuracy           : {log_accuracy:.4f}\n")
        file.write(f"Cross-Validation Accuracy (mean) : {np.mean(log_cv_scores):.4f}\n\n")

        file.write(
            "Catatan: Random Forest biasanya memberikan performa lebih baik, "
            "sedangkan Logistic Regression berfungsi sebagai baseline.\n"
            "Cross-validation membantu menilai performa rata-rata model.\n"
        )

    # Simpan confusion matrix ke CSV supaya bisa divisualisasikan
    cm_df = pd.DataFrame(
        rf_cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )
    cm_df.to_csv(REPORTS_DIR / "classification_confusion_matrix.csv", index=True)

    print("Evaluasi klasifikasi selesai, model disimpan.")