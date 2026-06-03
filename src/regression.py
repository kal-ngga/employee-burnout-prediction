import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import (
    CLUSTERING_RESULT_PATH,
    REGRESSION_MODEL_PATH,
    REGRESSION_EVALUATION_PATH,
    REPORTS_DIR,
    NUMERIC_FEATURES,
    create_directories
)

def run_regression():
    create_directories()
    df = pd.read_csv(CLUSTERING_RESULT_PATH)

    X = df[NUMERIC_FEATURES]
    y = df["Cluster"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Random Forest Regressor
    rf_reg = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    rf_reg.fit(X_train, y_train)
    y_pred_rf = rf_reg.predict(X_test)

    # 2. Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)

    # K-fold cross‑validation (misal cv=5) untuk kedua model
    rf_cv_scores = cross_val_score(rf_reg, X, y, cv=5, scoring="r2")
    lin_cv_scores = cross_val_score(lin_reg, X, y, cv=5, scoring="r2")


    # Evaluasi pada test set
    def evaluate(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    rf_mae, rf_rmse, rf_r2 = evaluate(y_test, y_pred_rf) #RandomForestRegressor
    lin_mae, lin_rmse, lin_r2 = evaluate(y_test, y_pred_lin) #LinearRegression
    joblib.dump(rf_reg, REGRESSION_MODEL_PATH)

    with open(REGRESSION_EVALUATION_PATH, "w") as file:
        file.write("REGRESSION EVALUATION\n")
        file.write("=====================\n\n")
        file.write("Model 1: RandomForestRegressor\n")
        file.write(f"MAE  : {rf_mae:.4f}\n")
        file.write(f"RMSE : {rf_rmse:.4f}\n")
        file.write(f"R2   : {rf_r2:.4f}\n")
        file.write(f"Cross-Validation R2 (mean of 5 folds): {np.mean(rf_cv_scores):.4f}\n\n")

        file.write("Model 2: LinearRegression\n")
        file.write(f"MAE  : {lin_mae:.4f}\n")
        file.write(f"RMSE : {lin_rmse:.4f}\n")
        file.write(f"R2   : {lin_r2:.4f}\n")
        file.write(f"Cross-Validation R2 (mean of 5 folds): {np.mean(lin_cv_scores):.4f}\n\n")

        file.write(
            "Catatan: Cross-validation membantu menilai kestabilan model dengan data subset "
            "yang berbeda-beda, sehingga hasil evaluasi lebih andal."
        )

    print("Evaluasi regresi selesai, model disimpan.")