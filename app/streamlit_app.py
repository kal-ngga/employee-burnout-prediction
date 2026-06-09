"""Aplikasi Streamlit untuk Employee Burnout Prediction."""

import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

# Jika streamlit_app.py ada di root project
if (CURRENT_DIR / "src").exists():
    ROOT_DIR = CURRENT_DIR
# Jika streamlit_app.py ada di folder app/
else:
    ROOT_DIR = CURRENT_DIR.parent

MPL_CACHE_DIR = ROOT_DIR / "assets" / "matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from utils import (  # noqa: E402
    CLASSIFICATION_EVALUATION_PATH,
    CLASSIFICATION_MODEL_PATH,
    CLUSTER_RISK_MAPPING_PATH,
    CLUSTER_SUMMARY_PATH,
    DATA_IDENTIFICATION_REPORT_PATH,
    KMEANS_PATH,
    NUMERIC_FEATURES,
    REGRESSION_EVALUATION_PATH,
    REPORTS_DIR,
    SCALER_PATH,
    SILHOUETTE_SCORES_PATH,
    load_model,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_risk_mapping():
    """Membaca file cluster_risk_mapping.csv."""
    mapping_df = pd.read_csv(CLUSTER_RISK_MAPPING_PATH)

    risk_mapping = {}

    for _, row in mapping_df.iterrows():
        cluster = int(row["Cluster"])
        risk_mapping[cluster] = {
            "risk": row["Risk Label"],
            "recommendation": row["Recommendation"],
        }

    return risk_mapping


def show_image_if_exists(filename, caption):
    """Menampilkan gambar jika file ditemukan."""
    possible_paths = [
        ROOT_DIR / filename,
        ROOT_DIR / "assets" / filename,
    ]

    for path in possible_paths:
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")
            return True

    st.warning(f"Visualisasi `{filename}` belum tersedia. Jalankan `python src/main.py`.")
    return False


def read_csv_report(path):
    """Membaca report CSV dan menampilkan pesan jika file belum tersedia."""
    if not path.exists():
        st.warning(f"Report `{path.name}` belum tersedia. Jalankan `python src/main.py`.")
        return None

    try:
        return pd.read_csv(path)
    except Exception as error:
        st.error(f"Gagal membaca `{path.name}`: {error}")
        return None


def read_text_report(path):
    """Membaca report teks dan menampilkan pesan jika file belum tersedia."""
    if not path.exists():
        st.warning(f"Report `{path.name}` belum tersedia. Jalankan `python src/main.py`.")
        return None

    try:
        return path.read_text(encoding="utf-8")
    except Exception as error:
        st.error(f"Gagal membaca `{path.name}`: {error}")
        return None


def parse_regression_metrics(report_text):
    """Mengubah report evaluasi regresi menjadi dataframe."""
    pattern = re.compile(
        r"Model \d+: (?P<Model>[^\n]+)\n"
        r"MAE\s*:\s*(?P<MAE>[-\d.]+)\n"
        r"RMSE\s*:\s*(?P<RMSE>[-\d.]+)\n"
        r"R2\s*:\s*(?P<R2>[-\d.]+)\n"
        r"Cross-Validation R2 \(mean of 5 folds\): (?P<Cross_Validation_R2>[-\d.]+)"
    )
    return pd.DataFrame(match.groupdict() for match in pattern.finditer(report_text))


def parse_classification_metrics(report_text):
    """Mengubah report evaluasi klasifikasi menjadi dataframe."""
    pattern = re.compile(
        r"Model \d+: (?P<Model>[^\n]+)\n"
        r"Test Accuracy\s*:\s*(?P<Test_Accuracy>[\d.]+)\n"
        r"Cross-Validation Accuracy \(mean\)\s*:\s*"
        r"(?P<Cross_Validation_Accuracy>[\d.]+)"
    )
    return pd.DataFrame(match.groupdict() for match in pattern.finditer(report_text))


def render_data_preparation_findings():
    """Menampilkan ringkasan sederhana data preparation."""
    st.markdown("### Tujuan")
    st.write(
        "Menyiapkan data karyawan agar bersih dan siap digunakan oleh model. "
        "Missing value ditangani dan fitur `Years Working` dibuat dari tanggal bergabung."
    )

    report = read_csv_report(DATA_IDENTIFICATION_REPORT_PATH)
    if report is not None:
        variable_report = report[report["data_nature"] != "summary"].copy()
        total_records = report.loc[
            report["variable_name"] == "TOTAL_RECORDS", "data_type"
        ].iloc[0]
        total_variables = report.loc[
            report["variable_name"] == "TOTAL_VARIABLES", "data_type"
        ].iloc[0]
        total_missing = pd.to_numeric(
            variable_report["missing_values"], errors="coerce"
        ).sum()

        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Total Record Awal", f"{int(float(total_records)):,}")
        metric_2.metric("Total Variabel Awal", int(float(total_variables)))
        metric_3.metric("Missing Value Awal", f"{int(total_missing):,}")

        missing_values = variable_report[["variable_name", "missing_values"]].copy()
        missing_values["missing_values"] = pd.to_numeric(
            missing_values["missing_values"], errors="coerce"
        ).fillna(0)
        missing_values = missing_values.set_index("variable_name")
        st.markdown("#### Kondisi Data Awal")
        st.bar_chart(missing_values)

        missing_columns = missing_values[missing_values["missing_values"] > 0]
        st.info(
            "Missing value ditemukan pada "
            f"**{', '.join(missing_columns.index)}** dan sudah ditangani sebelum modeling."
        )

    with st.expander("Lihat korelasi antarvariabel"):
        show_image_if_exists("heatmap.png", "Correlation Heatmap")
        st.caption(
            "Mental Fatigue Score dan Resource Allocation memiliki hubungan yang cukup "
            "jelas dengan Burn Rate."
        )

    with st.expander("Lihat distribusi dan boxplot variabel"):
        show_image_if_exists(
            "numeric_distributions.png",
            "Distribusi Seluruh Variabel Numerik",
        )
        st.caption(
            "Histogram menunjukkan pola sebaran nilai pada setiap variabel numerik."
        )
        show_image_if_exists(
            "numeric_boxplots.png",
            "Boxplot Seluruh Variabel Numerik",
        )
        st.caption(
            "Boxplot membantu melihat rentang nilai dan kemungkinan nilai ekstrem."
        )


def render_clustering_findings():
    """Menampilkan ringkasan sederhana clustering."""
    st.markdown("### Tujuan")
    st.write(
        "Mengelompokkan karyawan berdasarkan kemiripan karakteristiknya menggunakan "
        "K-Means. Hasil evaluasi menunjukkan bahwa **2 cluster** merupakan pilihan terbaik."
    )

    silhouette_scores = read_csv_report(SILHOUETTE_SCORES_PATH)
    if silhouette_scores is not None:
        best_row = silhouette_scores.loc[
            silhouette_scores["Silhouette Score"].idxmax()
        ]

        metric_1, metric_2 = st.columns(2)
        metric_1.metric("Jumlah Cluster", "2")
        metric_2.metric("Silhouette Score", f"{best_row['Silhouette Score']:.3f}")

    summary = read_csv_report(CLUSTER_SUMMARY_PATH)
    if summary is not None:
        high_risk = summary.loc[summary["Avg_Burn_Rate"].idxmax()]
        low_risk = summary.loc[summary["Avg_Burn_Rate"].idxmin()]
        st.info(
            f"Cluster {int(low_risk['Cluster'])} adalah **Low Burnout Risk**, sedangkan "
            f"Cluster {int(high_risk['Cluster'])} adalah **High Burnout Risk**. Kelompok "
            "berisiko tinggi memiliki beban kerja dan kelelahan mental yang lebih besar."
        )

    pca_col, distribution_col = st.columns(2)
    with pca_col:
        show_image_if_exists("pca_clustering.png", "PCA Clustering Visualization")
        st.caption("Pemisahan dua kelompok karyawan.")
    with distribution_col:
        show_image_if_exists("distribution.png", "Burn Rate Distribution by Cluster")
        st.caption("Perbandingan Burn Rate pada setiap kelompok.")

    with st.expander("Lihat alasan pemilihan 2 cluster"):
        elbow_col, silhouette_col = st.columns(2)
        with elbow_col:
            show_image_if_exists("elbow_method.png", "Elbow Method")
        with silhouette_col:
            show_image_if_exists("silhouette_score.png", "Silhouette Score")


def render_regression_findings():
    """Menampilkan ringkasan sederhana evaluasi regresi."""
    st.markdown("### Tujuan")
    st.write(
        "Membandingkan **Random Forest Regressor** dan **Linear Regression** untuk "
        "memprediksi nilai **Burn Rate** sebagai target kontinu."
    )

    report_text = read_text_report(REGRESSION_EVALUATION_PATH)
    if report_text is None:
        return

    metrics = parse_regression_metrics(report_text)
    if metrics.empty:
        st.warning("Metrik regresi tidak dapat dibaca dari report.")
        return

    best_model = metrics.loc[metrics["R2"].astype(float).idxmax()]
    model_1, model_2 = metrics.iloc[0], metrics.iloc[1]
    metric_1, metric_2 = st.columns(2)
    metric_1.metric(f"R² {model_1['Model']}", f"{float(model_1['R2']):.4f}")
    metric_2.metric(f"R² {model_2['Model']}", f"{float(model_2['R2']):.4f}")

    chart_df = metrics.set_index("Model")[["R2", "Cross_Validation_R2"]].astype(float)
    st.bar_chart(chart_df)
    st.info(
        f"**{best_model['Model']}** memberikan hasil terbaik dengan nilai R² "
        f"**{float(best_model['R2']):.4f}**."
    )
    st.caption(
        "Evaluasi ini tidak menggunakan Cluster sebagai target regresi, sehingga hasilnya "
        "tidak lagi sempurna karena menyalin label hasil clustering."
    )


def render_classification_findings():
    """Menampilkan ringkasan sederhana evaluasi klasifikasi."""
    st.markdown("### Tujuan")
    st.write(
        "Membandingkan **Random Forest Classifier** dan **Logistic Regression** untuk "
        "memprediksi **Low** atau **High Burnout Risk** dari target turunan Burn Rate."
    )

    report_text = read_text_report(CLASSIFICATION_EVALUATION_PATH)
    if report_text is not None:
        metrics = parse_classification_metrics(report_text)
        if not metrics.empty:
            model_1, model_2 = metrics.iloc[0], metrics.iloc[1]
            metric_1, metric_2 = st.columns(2)
            metric_1.metric(
                f"Accuracy {model_1['Model']}",
                f"{float(model_1['Test_Accuracy']) * 100:.2f}%",
            )
            metric_2.metric(
                f"Accuracy {model_2['Model']}",
                f"{float(model_2['Test_Accuracy']) * 100:.2f}%",
            )
            st.info(
                "Model utama pada fitur prediksi aplikasi adalah **Random Forest "
                "Classifier**. Label yang diprediksi adalah kelas risiko dari Burn Rate, "
                "bukan Cluster hasil K-Means."
            )

    confusion_matrix_path = REPORTS_DIR / "classification_confusion_matrix.csv"
    confusion_matrix = read_csv_report(confusion_matrix_path)
    if confusion_matrix is not None:
        confusion_matrix = confusion_matrix.rename(
            columns={confusion_matrix.columns[0]: "Actual"}
        ).set_index("Actual")
        correct_predictions = sum(
            confusion_matrix.iloc[index, index]
            for index in range(min(confusion_matrix.shape))
        )
        total_predictions = confusion_matrix.to_numpy().sum()
        st.write(
            f"Confusion matrix menunjukkan **{correct_predictions:,} dari "
            f"{total_predictions:,}** data uji diprediksi pada cluster yang benar."
        )
        with st.expander("Lihat confusion matrix"):
            st.dataframe(
                confusion_matrix.style.background_gradient(cmap="Blues"),
                width="stretch",
            )


# ============================================================
# TAB 1 - PREDICTION
# ============================================================

def render_prediction_tab(classifier, kmeans, scaler, risk_mapping):
    """Menampilkan tab prediksi burnout."""
    st.subheader("Prediksi Risiko Burnout Karyawan")

    st.markdown(
        """
        Isi data karyawan pada form di bawah ini.

        Aplikasi akan memprediksi apakah karyawan termasuk ke dalam kelompok
        **Low Burnout Risk** atau **High Burnout Risk** berdasarkan model yang sudah dilatih.

        **Penjelasan Variabel Input:**

        - **Designation**
          Level jabatan karyawan. Nilai berada pada rentang **0 sampai 5**.
          Semakin tinggi nilainya, semakin tinggi level jabatan.

        - **Resource Allocation**
          Besarnya beban kerja atau alokasi resource karyawan. Nilai berada pada rentang **1 sampai 10**.
          Semakin tinggi nilainya, semakin besar beban kerja.

        - **Mental Fatigue Score**
          Tingkat kelelahan mental karyawan. Nilai berada pada rentang **0 sampai 10**.
          Nilai 0 berarti tidak lelah, sedangkan 10 berarti sangat lelah.

        - **Years Working**
          Lama karyawan bekerja di perusahaan dalam satuan tahun.
        """
    )
    st.markdown("-" * 20)

    col1, col2 = st.columns(2)

    with col1:
        designation = st.number_input(
            "Designation (0 - 5)",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=1.0,
        )

        resource_allocation = st.number_input(
            "Resource Allocation (1 - 10)",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=1.0,
        )

    with col2:
        mental_fatigue_score = st.slider(
            "Mental Fatigue Score (0 - 10)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
        )

        years_working = st.number_input(
            "Years Working",
            min_value=0.0,
            max_value=50.0,
            value=3.0,
            step=1.0,
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

        predicted_risk_class = int(classifier.predict(input_df)[0])
        kmeans_cluster = int(kmeans.predict(scaler.transform(input_df))[0])

        risk_info_by_class = {
            0: {
                "risk": "Low Burnout Risk",
                "recommendation": "Pertahankan beban kerja dan pantau kondisi karyawan secara berkala.",
            },
            1: {
                "risk": "High Burnout Risk",
                "recommendation": "Evaluasi beban kerja, alokasi resource, dan dukungan pemulihan karyawan.",
            },
        }
        risk_info = risk_info_by_class.get(
            predicted_risk_class,
            {
                "risk": "Unknown Risk",
                "recommendation": "Jalankan ulang pipeline training untuk memperbarui model klasifikasi.",
            },
        )
        cluster_info = risk_mapping.get(kmeans_cluster)

        st.markdown("### Hasil Prediksi")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric("Predicted Risk Class", "High" if predicted_risk_class == 1 else "Low")

        with result_col2:
            st.metric("Burnout Risk", risk_info["risk"])

        st.write(f"**Recommendation:** {risk_info['recommendation']}")
        if cluster_info is not None:
            st.caption(
                "K-Means reference from the same input: "
                f"Cluster {kmeans_cluster} ({cluster_info['risk']})."
            )
        else:
            st.caption(f"K-Means reference from the same input: Cluster {kmeans_cluster}.")


# ============================================================
# TAB 2 - PROJECT INSIGHT
# ============================================================

def render_insight_tab():
    """Menampilkan seluruh proses dan temuan proyek secara berurutan."""
    st.subheader("Cerita di Balik Project")

    st.markdown(
        """
        Project ini bertujuan mengelompokkan risiko burnout karyawan dan membuat model
        sederhana untuk memprediksi kelompok risiko tersebut.

        Alurnya: **menyiapkan data → membuat cluster → membandingkan model → menampilkan
        hasil prediksi**.
        """
    )

    preparation_tab, clustering_tab, regression_tab, classification_tab = st.tabs(
        [
            "1. Data Preparation",
            "2. Clustering",
            "3. Regression",
            "4. Classification",
        ]
    )

    with preparation_tab:
        render_data_preparation_findings()

    with clustering_tab:
        render_clustering_findings()

    with regression_tab:
        render_regression_findings()

    with classification_tab:
        render_classification_findings()

    st.success(
        "Hasil akhir: Random Forest Classifier digunakan pada tab **Prediksi Burnout** "
        "untuk memprediksi Low atau High Burnout Risk dan memberikan rekomendasi sederhana."
    )


# ============================================================
# MAIN APP
# ============================================================

def main():
    """Menjalankan aplikasi Streamlit."""
    st.set_page_config(
        page_title="Employee Burnout Prediction",
        layout="wide",
    )

    st.title("Employee Burnout Prediction")
    st.write(
        "Aplikasi sederhana untuk memprediksi risiko burnout karyawan "
        "dan menampilkan temuan utama dari proses data mining."
    )

    try:
        scaler = load_model(SCALER_PATH)
        kmeans = load_model(KMEANS_PATH)
        classifier = load_model(CLASSIFICATION_MODEL_PATH)
        risk_mapping = load_risk_mapping()

    except FileNotFoundError as error:
        st.error(
            f"{error}\n\n"
            "File model atau report belum ditemukan. "
            "Jalankan dulu pipeline dari root project dengan command:\n\n"
            "`python src/main.py`"
        )
        st.stop()

    tab_prediction, tab_insight = st.tabs(
        [
            "Prediksi Burnout",
            "Temuan Proyek",
        ]
    )

    with tab_prediction:
        render_prediction_tab(classifier, kmeans, scaler, risk_mapping)

    with tab_insight:
        render_insight_tab()


if __name__ == "__main__":
    main()
