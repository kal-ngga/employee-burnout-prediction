import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from utils import (
    CLEANED_DATA_PATH,
    CLASSIFICATION_MODEL_PATH,
    CLASSIFICATION_EVALUATION_PATH,
    REPORTS_DIR,
    NUMERIC_FEATURES,
    create_directories
)

def run_classification():
    create_directories()
    df = pd.read_csv(CLEANED_DATA_PATH)

    X = df[NUMERIC_FEATURES]
    burn_rate_threshold = df["Burn Rate"].median()
    y = (df["Burn Rate"] >= burn_rate_threshold).astype(int)

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
    log_reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)

    rf_cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=5, scoring="accuracy")
    log_cv_scores = cross_val_score(log_reg, X_train, y_train, cv=5, scoring="accuracy")

    # Evaluasi pada test set
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    log_accuracy = accuracy_score(y_test, y_pred_log)

    # Confusion matrix dan classification report untuk model Random Forest
    rf_cm = confusion_matrix(y_test, y_pred_rf)
    target_names = ["Low Burnout Risk", "High Burnout Risk"]
    rf_report = classification_report(y_test, y_pred_rf, target_names=target_names)

    # Simpan model Random Forest sebagai model utama
    joblib.dump(rf_clf, CLASSIFICATION_MODEL_PATH)

    with open(CLASSIFICATION_EVALUATION_PATH, "w") as file:
        file.write("CLASSIFICATION EVALUATION\n")
        file.write("=========================\n\n")
        file.write("Target: Burnout Risk Class\n")
        file.write(f"Rule: High Burnout Risk jika Burn Rate >= {burn_rate_threshold:.4f}\n")
        file.write(f"Features: {', '.join(NUMERIC_FEATURES)}\n")
        file.write(
            "Leakage control: Burn Rate, Cluster, dan label turunan target tidak digunakan "
            "sebagai fitur input.\n\n"
        )
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
            "Catatan: Klasifikasi sekarang memprediksi kelas risiko dari Burn Rate, bukan "
            "menyalin label Cluster hasil K-Means. Cross-validation dihitung pada data "
            "train untuk mengurangi bias evaluasi.\n"
        )

    # Simpan confusion matrix ke CSV supaya bisa divisualisasikan
    cm_df = pd.DataFrame(
        rf_cm,
        index=["Actual Low Burnout Risk", "Actual High Burnout Risk"],
        columns=["Predicted Low Burnout Risk", "Predicted High Burnout Risk"]
    )
    cm_df.to_csv(REPORTS_DIR / "classification_confusion_matrix.csv", index=True)

    print("Evaluasi klasifikasi selesai, model disimpan.")
