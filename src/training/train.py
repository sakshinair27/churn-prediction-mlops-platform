"""
Training entrypoint for the churn classifier.

- Loads / generates the churn dataset
- Builds a preprocessing + GradientBoostingClassifier pipeline
- Runs a small hyperparameter sweep, logging every run to MLflow
  (params, metrics, model artifact, confusion matrix, feature importance)
- Registers the best run's model in the MLflow Model Registry as
  "churn-predictor" and promotes it to the "Production" alias/stage
- Also saves the winning pipeline to models/model.pkl for the serving
  layer to load without requiring a live MLflow tracking server.
"""
import argparse
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.generate_data import generate_churn_dataset  # noqa: E402

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
    "num_additional_services",
    "avg_monthly_usage_gb",
]
BINARY_FEATURES = [
    "has_tech_support",
    "has_online_security",
    "paperless_billing",
    "senior_citizen",
    "partner",
    "dependents",
]
CATEGORICAL_FEATURES = ["contract_type", "internet_service", "payment_method"]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"

MODEL_NAME = "churn-predictor"
EXPERIMENT_NAME = "customer-churn-prediction"


def build_pipeline(**gbc_kwargs) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    clf = GradientBoostingClassifier(random_state=42, **gbc_kwargs)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }, y_pred, y_proba


def log_confusion_matrix(y_test, y_pred, run_dir: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    path = os.path.join(run_dir, "confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Train the churn classifier")
    parser.add_argument(
        "--data-path",
        default="data/customer_churn.csv",
        help="Path to the training CSV (default: synthetic dataset). "
             "Pass data/customer_churn_real.csv to train on the real "
             "IBM Telco Customer Churn dataset instead.",
    )
    parser.add_argument(
        "--experiment-name",
        default=EXPERIMENT_NAME,
        help="MLflow experiment name to log runs under.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    data_path = args.data_path
    if not os.path.exists(data_path):
        if data_path == "data/customer_churn.csv":
            df = generate_churn_dataset()
            df.to_csv(data_path, index=False)
        else:
            raise FileNotFoundError(
                f"{data_path} not found. Run "
                f"`python src/training/prepare_real_data.py` first if you "
                f"intended to train on the real Telco dataset."
            )
    else:
        df = pd.read_csv(data_path)

    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Small hyperparameter sweep -- every configuration is a tracked MLflow
    # run. class_weight-style rebalancing (via sample_weight) is included
    # since real churn data is imbalanced (~26% positive class) and
    # unweighted GBMs under-predict the minority (churn) class.
    search_space = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "balanced": True},
        {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03, "balanced": True},
    ]

    # Model-selection metric: F1 rather than raw ROC-AUC. For churn
    # retention, a missed churner (false negative) is typically far more
    # costly than an unnecessary retention offer to a loyal customer
    # (false positive) -- so we optimize for a balance of precision and
    # recall on the churn class rather than pure separability.
    SELECTION_METRIC = "f1_score"

    best_run = None
    best_score = -np.inf
    best_pipeline = None
    best_metrics = None
    all_results = []

    client = MlflowClient()

    for i, raw_params in enumerate(search_space):
        params = dict(raw_params)
        balanced = params.pop("balanced", False)
        with mlflow.start_run(run_name=f"gbc-run-{i}") as run:
            start = time.time()
            pipeline = build_pipeline(**params)

            if balanced:
                # GradientBoostingClassifier has no native class_weight, so
                # rebalancing is done via sample_weight on the minority
                # (churn) class -- proportional to the real class imbalance.
                pos_rate = y_train.mean()
                weight_pos = (1 - pos_rate) / pos_rate
                sample_weight = np.where(y_train == 1, weight_pos, 1.0)
                pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weight)
            else:
                pipeline.fit(X_train, y_train)
            train_time = time.time() - start

            metrics, y_pred, y_proba = evaluate(pipeline, X_test, y_test)

            mlflow.log_params(params)
            mlflow.log_param("balanced", balanced)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))
            mlflow.log_metrics(metrics)
            mlflow.log_metric("train_time_seconds", train_time)

            cm_path = log_confusion_matrix(y_test, y_pred, run_dir="reports")
            mlflow.log_artifact(cm_path, artifact_path="plots")

            signature = infer_signature(X_train, pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(3),
            )

            print(f"[run {i}] params={raw_params} metrics={metrics}")
            all_results.append({"run_id": run.info.run_id, "params": raw_params, "metrics": metrics})

            if metrics[SELECTION_METRIC] > best_score:
                best_score = metrics[SELECTION_METRIC]
                best_run = run
                best_pipeline = pipeline
                best_metrics = metrics

    print(f"\nBest run: {best_run.info.run_id} ({SELECTION_METRIC}={best_score:.4f})")

    # Register the best model in the MLflow Model Registry.
    model_uri = f"runs:/{best_run.info.run_id}/model"
    try:
        registered = mlflow.register_model(model_uri, MODEL_NAME)
        # Promote to Production stage (classic registry API; alias-based
        # registries would use set_registered_model_alias instead).
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=registered.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Registered {MODEL_NAME} v{registered.version} -> Production")
    except Exception as e:  # pragma: no cover - registry may be unavailable in some envs
        print(f"Model registry step skipped/failed (non-fatal): {e}")

    # Save the winning pipeline locally for the FastAPI serving layer.
    joblib.dump(best_pipeline, "models/model.pkl")
    with open("models/feature_schema.json", "w") as f:
        json.dump(
            {
                "numeric_features": NUMERIC_FEATURES,
                "binary_features": BINARY_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "categorical_values": {
                    "contract_type": sorted(df["contract_type"].unique().tolist()),
                    "internet_service": sorted(df["internet_service"].unique().tolist()),
                    "payment_method": sorted(df["payment_method"].unique().tolist()),
                },
            },
            f,
            indent=2,
        )

    with open("reports/training_metrics.json", "w") as f:
        json.dump(
            {
                "best_run_id": best_run.info.run_id,
                "best_params": search_space[[r["run_id"] for r in all_results].index(best_run.info.run_id)],
                "best_metrics": best_metrics,
                "all_runs": all_results,
                "model_name": MODEL_NAME,
            },
            f,
            indent=2,
        )

    print("\n=== FINAL MODEL METRICS (held-out test set) ===")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nSaved model -> models/model.pkl")
    print("Saved metrics -> reports/training_metrics.json")


if __name__ == "__main__":
    main()
