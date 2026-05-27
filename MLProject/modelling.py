from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "breast_cancer_preprocessing" / "breast_cancer_preprocessed.csv"
MLRUNS_DIR = ROOT_DIR / "mlruns"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
TARGET_COLUMN = "diagnosis"
EXPERIMENT_NAME = "Workflow CI - Breast Cancer Classification"
RANDOM_STATE = 42


def load_dataset(data_path: Path = DATA_PATH):
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset hasil preprocessing tidak ditemukan: {data_path}")

    data = pd.read_csv(data_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Kolom target '{TARGET_COLUMN}' tidak ditemukan pada dataset.")

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN].astype(int)
    return X, y, data


def save_artifacts(y_test, y_pred, y_proba, model, feature_names):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_path = ARTIFACT_DIR / "classification_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    metric_path = ARTIFACT_DIR / "metric_info.json"
    with open(metric_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm)
    ax.set_title("Training Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    cm_path = ARTIFACT_DIR / "training_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        importance_path = ARTIFACT_DIR / "feature_importance.csv"
        importance.to_csv(importance_path, index=False)

    prediction_path = ARTIFACT_DIR / "test_predictions.csv"
    pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred,
            "y_probability": y_proba,
        }
    ).to_csv(prediction_path, index=False)

    return metrics


def train_model():
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_models=True)

    X, y, data = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="workflow_ci_random_forest") as run:
        mlflow.log_param("dataset_path", str(DATA_PATH))
        mlflow.log_param("dataset_shape", data.shape)
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("test_size", 0.2)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = save_artifacts(
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            model=model,
            feature_names=X.columns.tolist(),
        )

        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(ARTIFACT_DIR), artifact_path="training_artifacts")
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        latest_run_path = ROOT_DIR / "latest_run_id.txt"
        latest_run_path.write_text(run_id, encoding="utf-8")

        print("Training selesai.")
        print(f"Experiment name: {EXPERIMENT_NAME}")
        print(f"Tracking URI: file:{MLRUNS_DIR}")
        print(f"Run ID: {run_id}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    train_model()
