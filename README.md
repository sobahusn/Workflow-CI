# Adult Income - MLflow CI Workflow

Proyek ini menjalankan training model Random Forest untuk klasifikasi pendapatan (Adult Income dataset) menggunakan **MLflow Projects** dengan pencatatan otomatis via `mlflow.sklearn.autolog`.

## Struktur Proyek

```
Workflow-CI/
├── MLProject/
│   ├── MLProject           # Konfigurasi entry point MLflow
│   ├── conda.yaml          # Spesifikasi environment
│   ├── modelling.py        # Script training utama
│   └── adult_preprocessing/
│       ├── train_preprocessed.csv
│       └── test_preprocessed.csv
└── mlruns/                 # Hasil run MLflow (auto-generated)
```

## Cara Menjalankan

### 1. Siapkan virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install mlflow==2.19.0 pandas scikit-learn numpy matplotlib joblib
```

### 2. Jalankan MLflow Project

```bash
mlflow run MLProject --env-manager=local
```

`--env-manager=local` berarti MLflow menggunakan environment Python yang sedang aktif (venv) tanpa membuat conda environment baru.

### 3. Lihat hasil di MLflow UI

Buka browser ke `http://localhost:5000` untuk melihat metrics, parameter, dan artifacts.
