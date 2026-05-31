from pathlib import Path
import json
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "adult_preprocessing"
TRAIN_PATH = DATA_DIR / "train_preprocessed.csv"
TEST_PATH = DATA_DIR / "test_preprocessed.csv"
TARGET_COLUMN = "income"


def load_dataset():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "File train_preprocessed.csv dan test_preprocessed.csv belum ditemukan. "
            "Copy hasil preprocessing dari Kriteria 1 ke folder MLProject/adult_preprocessing/."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = load_dataset()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    # Menggunakan autolog.
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    with mlflow.start_run(run_name="basic_random_forest_autolog_CI"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_pred, zero_division=0),
            "test_f1_score": f1_score(y_test, y_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, y_proba),
        }

        # Menambahkan metric manual agar mudah dibaca di dashboard.
        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))

        print("Training selesai. Metrics:")
        print(json.dumps(metrics, indent=4))
        print("Model dan metrics sudah tercatat di MLflow.")


if __name__ == "__main__":
    main()
