"""Aplikasi Streamlit untuk prediksi burnout."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils import (  # noqa: E402
    CLASSIFICATION_MODEL_PATH,
    CLUSTER_RISK_MAPPING_PATH,
    KMEANS_PATH,
    NUMERIC_FEATURES,
    SCALER_PATH,
    load_model,
)

def load_risk_mapping():
    """Membaca keterangan risiko dari file cluster_risk_mapping.csv."""
    mapping_df = pd.read_csv(CLUSTER_RISK_MAPPING_PATH)

    risk_mapping = {}
    for _, row in mapping_df.iterrows():
        cluster = int(row["Cluster"])
        risk_mapping[cluster] = {
            "risk": row["Risk Label"],
            "recommendation": row["Recommendation"],
        }

    return risk_mapping


def main():
    """Menampilkan form dan hasil prediksi."""
    st.set_page_config(page_title="Employee Burnout Prediction")
    st.title("Employee Burnout Prediction")

    try:
        scaler = load_model(SCALER_PATH)
        kmeans = load_model(KMEANS_PATH)
        classifier = load_model(CLASSIFICATION_MODEL_PATH)
        risk_mapping = load_risk_mapping()
    except FileNotFoundError as error:
        st.error(f"{error}. Run `python src/main.py` from the project root first.")
        st.stop()

    designation = st.number_input(
        "Designation", min_value=0.0, max_value=5.0, value=2.0, step=1.0
    )
    resource_allocation = st.number_input(
        "Resource Allocation", min_value=1.0, max_value=10.0, value=5.0, step=1.0
    )
    mental_fatigue_score = st.slider(
        "Mental Fatigue Score", 0.0, 10.0, 5.0, 0.1
    )
    years_working = st.number_input(
        "Years Working", min_value=0.0, max_value=50.0, value=3.0, step=1.0
    )

    if st.button("Predict Burnout Risk", type="primary"):
        input_df = pd.DataFrame(
            [
                {
                    "Designation": designation,
                    "Resource Allocation": resource_allocation,
                    "Mental Fatigue Score": mental_fatigue_score,
                    "Years Working": years_working,
                }
            ],
            columns=NUMERIC_FEATURES,
        )

        predicted_cluster = int(classifier.predict(input_df)[0])
        kmeans_cluster = int(kmeans.predict(scaler.transform(input_df))[0])
        risk_info = risk_mapping.get(
            predicted_cluster,
            {
                "risk": "Unknown Risk",
                "recommendation": "Run the training pipeline again to refresh cluster mapping.",
            },
        )

        st.subheader(f"Predicted cluster: {predicted_cluster}")
        st.metric("Burnout risk", risk_info["risk"])
        st.write(f"Recommendation: {risk_info['recommendation']}")
        st.caption(f"K-Means reference cluster from the same input: {kmeans_cluster}")


if __name__ == "__main__":
    main()
