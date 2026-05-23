# Employee Burnout Prediction

Project portfolio Data Mining untuk memprediksi kelompok risiko burnout karyawan menggunakan framework CRISP-DM.

## Dataset

Dataset asli disimpan di:

```text
data/raw/employee_burnout_analysis.csv
```

Kolom dataset:

- Employee ID
- Date of Joining
- Gender
- Company Type
- WFH Setup Available
- Designation
- Resource Allocation
- Mental Fatigue Score
- Burn Rate

## Tahapan CRISP-DM

1. Business Understanding: memahami kebutuhan prediksi risiko burnout.
2. Data Understanding: membaca data, mengecek variabel, tipe data, record, dan missing values.
3. Data Preparation: membersihkan data, membuat `Years Working`, memilih fitur numerik.
4. Data Clustering: K-Means dengan 2 cluster dan menyimpan hasil ke `reports/clustering_result.csv`.
5. Regression: RandomForestRegressor untuk memprediksi label `Cluster`.
6. Classification: RandomForestClassifier untuk memprediksi label `Cluster`.
7. Evaluation: menyimpan evaluasi regression dan classification di folder `reports`.
8. Deployment: aplikasi Streamlit untuk prediksi risiko burnout.

## Cara Menjalankan

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan pipeline dari root project:

```bash
python src/main.py
```

Jalankan aplikasi Streamlit:

```bash
streamlit run app/streamlit_app.py
```

## Input Aplikasi

- Designation
- Resource Allocation
- Mental Fatigue Score
- Years Working

Output aplikasi:

- Predicted cluster
- Low Burnout Risk atau High Burnout Risk
- Recommendation sederhana

