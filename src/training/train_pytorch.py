"""
Trains a tabular neural network (MLP) in PyTorch on the same real
IBM Telco Customer Churn data, using the exact same preprocessing,
train/test split, and MLflow experiment as the sklearn GradientBoosting
sweep in train.py -- so the two model families are directly comparable
in the MLflow run table, not just "a model that happens to also run."

Usage:
    python src/training/train_pytorch.py --data-path data/customer_churn_real.csv \
        --experiment-name customer-churn-prediction-real
"""
import argparse
import json
import os
import time

# NOTE: torch must be imported before mlflow/sklearn on some Windows setups
# -- importing mlflow first can grab a conflicting copy of a shared DLL
# (observed as "OSError: [WinError 1114] DLL initialization routine failed"
# loading torch's c10.dll) that torch then fails to load cleanly.
import torch
import torch.nn as nn

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Must match train.py exactly so both model families see the same inputs.
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

EXPERIMENT_NAME = "customer-churn-prediction-real"
SEED = 42


class ChurnMLP(nn.Module):
    """A small feed-forward net for tabular churn classification.

    Architecture: input -> 64 -> 32 -> 1 (logit), with BatchNorm + Dropout
    for regularization -- standard sizing for a few thousand rows and a
    few dozen input features after one-hot encoding (too large a net would
    just overfit this dataset).
    """

    def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def evaluate(model, X_test_t, y_test, device) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        y_proba = torch.sigmoid(logits).cpu().numpy()
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }, y_pred, y_proba


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PyTorch MLP churn classifier")
    parser.add_argument("--data-path", default="data/customer_churn_real.csv")
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=True,
        help="Upweight the minority (churn) class via pos_weight in the loss, "
        "matching the sample_weight rebalancing used in the sklearn sweep.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    df = pd.read_csv(args.data_path)
    X = df[ALL_FEATURES]
    y = df[TARGET].values

    # Identical split to train.py: same test_size, random_state, and
    # stratify column, so both model families are evaluated on the exact
    # same held-out rows.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_np = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test_np = preprocessor.transform(X_test).astype(np.float32)
    if hasattr(X_train_np, "toarray"):  # OneHotEncoder can return sparse
        X_train_np = X_train_np.toarray()
        X_test_np = X_test_np.toarray()

    input_dim = X_train_np.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32)

    pos_rate = y_train.mean()
    pos_weight = torch.tensor((1 - pos_rate) / pos_rate, dtype=torch.float32) if args.balanced else None

    model = ChurnMLP(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    with mlflow.start_run(run_name="pytorch-mlp") as run:
        start = time.time()
        model.train()
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_ds)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  epoch {epoch + 1}/{args.epochs}  loss={epoch_loss:.4f}")
        train_time = time.time() - start

        metrics, y_pred, y_proba = evaluate(model, X_test_t, y_test, device)

        mlflow.log_params(
            {
                "architecture": "64-32-1 MLP, BatchNorm+Dropout(0.3)",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "optimizer": "Adam",
                "weight_decay": 1e-4,
                "balanced": args.balanced,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "input_dim": input_dim,
                "framework": "pytorch",
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_metric("train_time_seconds", train_time)

        # Log the raw PyTorch model. Note: unlike the sklearn pipeline in
        # train.py, this model expects already-preprocessed (scaled +
        # one-hot-encoded) float32 input -- the ColumnTransformer isn't
        # bundled into the PyTorch graph, so a serving layer would need to
        # apply `preprocessor.transform(...)` before calling this model.
        example_input = X_train_np[:3]
        signature = infer_signature(example_input, model(torch.tensor(example_input).to(device)).detach().cpu().numpy())
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=example_input,
        )

        print(f"\n[pytorch-mlp] metrics={metrics}")

        results = {
            "run_id": run.info.run_id,
            "framework": "pytorch",
            "architecture": "64-32-1 MLP",
            "metrics": metrics,
            "train_time_seconds": train_time,
        }

    report_path = "reports/training_metrics_pytorch.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== PYTORCH MLP METRICS (held-out test set) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved metrics -> {report_path}")


if __name__ == "__main__":
    main()
