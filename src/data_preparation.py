"""Bagian persiapan data."""

import pandas as pd
from datetime import datetime

from utils import (
    CLEANED_DATA_PATH,
    DATA_IDENTIFICATION_REPORT_PATH,
    NUMERIC_FEATURES,
    classify_column_type,
    read_raw_dataset,
)


def create_data_identification_report(df):
    """Membuat laporan sederhana tentang isi dataset."""
    report_data = []

    for column in df.columns:
        report_data.append(
            {
                "variable_name": column,
                "data_type": str(df[column].dtype),
                "data_nature": classify_column_type(df[column]),
                "missing_values": int(df[column].isna().sum()),
                "unique_values": int(df[column].nunique(dropna=True)),
            }
        )

    report = pd.DataFrame(report_data)

    summary_rows = pd.DataFrame(
        [
            {
                "variable_name": "TOTAL_VARIABLES",
                "data_type": len(df.columns),
                "data_nature": "summary",
                "missing_values": "",
                "unique_values": "",
            },
            {
                "variable_name": "TOTAL_RECORDS",
                "data_type": len(df),
                "data_nature": "summary",
                "missing_values": "",
                "unique_values": "",
            },
        ]
    )
    return pd.concat([report, summary_rows], ignore_index=True)


def prepare_data():
    """Membersihkan dataset dan menyimpannya ke folder data/processed."""
    df = read_raw_dataset()

    identification_report = create_data_identification_report(df)
    identification_report.to_csv(DATA_IDENTIFICATION_REPORT_PATH, index=False)

    df["Date of Joining"] = pd.to_datetime(
        df["Date of Joining"], format="%d/%m/%y", errors="coerce"
    )
    numeric_columns = [
        "Designation",
        "Resource Allocation",
        "Mental Fatigue Score",
        "Burn Rate",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    current_year = datetime.now().year
    df["Years Working"] = current_year - df["Date of Joining"].dt.year
    df["Years Working"] = df["Years Working"].fillna(0).astype(int)

    selected_columns = [
        "Employee ID",
        "Date of Joining",
        "Gender",
        "Company Type",
        "WFH Setup Available",
        "Designation",
        "Resource Allocation",
        "Mental Fatigue Score",
        "Years Working",
        "Burn Rate",
    ]
    cleaned_df = df[selected_columns].copy()

    for column in NUMERIC_FEATURES + ["Burn Rate"]:
        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())

    for column in ["Gender", "Company Type", "WFH Setup Available"]:
        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])

    cleaned_df = cleaned_df.dropna(subset=["Employee ID", "Date of Joining"]).reset_index(
        drop=True
    )
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)

    print("Business Understanding: Employee burnout risk will be grouped into two clusters.")
    print("Data Understanding: identification report saved to reports/data_identification_report.csv")
    print(f"Variable names: {list(df.columns)}")
    print(f"Total variables: {df.shape[1]}")
    print(f"Total records: {df.shape[0]}")
    print("Selected numeric variables for clustering, regression, and classification:")
    print(NUMERIC_FEATURES)
    print(f"Cleaned dataset saved to {CLEANED_DATA_PATH}")

    return cleaned_df


if __name__ == "__main__":
    prepare_data()
