import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import (
    CLEANED_DATA_PATH,
    REGRESSION_MODEL_PATH,
    REGRESSION_EVALUATION_PATH,
    NUMERIC_FEATURES,
    create_directories
)

def run_regression():
    create_directories()
    df = pd.read_csv(CLEANED_DATA_PATH)

    X = df[NUMERIC_FEATURES]
    y = df["Burn Rate"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv_scores = cross_val_score(rf_reg, X_train, y_train, cv=cv, scoring="r2")
    lin_cv_scores = cross_val_score(lin_reg, X_train, y_train, cv=cv, scoring="r2")


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
        file.write("Target: Burn Rate\n")
        file.write(f"Features: {', '.join(NUMERIC_FEATURES)}\n")
        file.write(
            "Leakage control: Burn Rate, Cluster, dan label turunan target tidak digunakan "
            "sebagai fitur input.\n\n"
        )
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
            "Catatan: Skor regresi tidak lagi mengevaluasi prediksi Cluster. Regresi "
            "sekarang memprediksi Burn Rate sebagai nilai kontinu, sehingga hasilnya "
            "lebih realistis untuk kasus burnout."
        )

    print("Evaluasi regresi selesai, model disimpan.")
