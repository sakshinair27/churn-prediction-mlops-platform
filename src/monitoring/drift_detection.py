"""
Production data-drift monitoring for the churn model, using Evidently AI.

What it does:
  1. Loads the original training distribution as the "reference" dataset.
  2. Simulates a batch of live production traffic that has drifted from
     that distribution (contract mix shifting more month-to-month, support
     tickets rising, usage climbing -- the kind of real-world shift you'd
     see during a pricing change or a competitor promo).
  3. Runs Evidently's DataDriftPreset to compare reference vs. current.
  4. Writes:
       reports/drift_report.html   - full interactive Evidently report
       reports/drift_summary.json  - machine-readable summary with an
                                      alert flag and a recommended action,
                                      suitable for wiring into CI/CD or an
                                      alerting pipeline.

This is meant to be run periodically (e.g. nightly, or triggered by a
sample of real scored traffic) against a fresh window of production
requests logged by the serving layer.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.generate_data import generate_churn_dataset  # noqa: E402

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
    "has_tech_support",
    "has_online_security",
    "paperless_billing",
    "senior_citizen",
    "partner",
    "dependents",
    "num_additional_services",
    "avg_monthly_usage_gb",
    "contract_type",
    "internet_service",
    "payment_method",
]

# Drift-severity thresholds used to translate Evidently's dataset-level
# drift share into an operational recommendation.
ALERT_THRESHOLD_SHARE_DRIFTED = 0.30  # >=30% of features drifted -> alert
RETRAIN_THRESHOLD_SHARE_DRIFTED = 0.50  # >=50% -> recommend retraining now


def simulate_production_traffic(
    n_samples: int = 2000, seed: int = 123, base_path: str = "data/customer_churn_real.csv"
) -> pd.DataFrame:
    """Simulate a window of live production traffic that has drifted from
    the training distribution: more month-to-month contracts, more support
    tickets, and heavier usage -- consistent with, e.g., a competitor
    promotion pulling in price-sensitive, high-usage customers.

    Important: this resamples and perturbs the SAME base dataset used as
    the drift-detection reference (the real Telco data when available),
    rather than drawing "current" traffic from a differently-generated
    source. Comparing two datasets built by different generators would
    register as "drift" purely from methodology differences, not from any
    real behavioral shift -- that would make the alert meaningless.
    """
    rng = np.random.default_rng(seed)
    if os.path.exists(base_path):
        base_df = pd.read_csv(base_path)
        df = base_df.sample(n=min(n_samples, len(base_df)), replace=len(base_df) < n_samples, random_state=seed).reset_index(drop=True)
    else:
        df = generate_churn_dataset(n_customers=n_samples, seed=seed)

    # Shift contract mix toward month-to-month.
    flip_mask = rng.random(n_samples) < 0.55
    df.loc[flip_mask, "contract_type"] = "Month-to-month"

    # Support tickets trending up sharply (e.g. a service-quality incident).
    df["num_support_tickets"] = (df["num_support_tickets"] + rng.poisson(3.2, n_samples)).clip(0, 25)

    # Usage climbing (e.g. more streaming/remote-work usage).
    df["avg_monthly_usage_gb"] = (df["avg_monthly_usage_gb"] * rng.normal(2.0, 0.2, n_samples)).clip(0, 2500)

    # Monthly charges creeping up (a price increase).
    df["monthly_charges"] = (df["monthly_charges"] * rng.normal(1.35, 0.1, n_samples)).clip(15, 350)

    # Tenure skewing younger (a surge of new signups from a promo).
    df["tenure_months"] = (df["tenure_months"] * rng.normal(0.55, 0.15, n_samples)).clip(0, 72).round().astype(int)

    # Payment method mix shifting toward electronic check.
    ec_mask = rng.random(n_samples) < 0.4
    df.loc[ec_mask, "payment_method"] = "Electronic check"

    return df


def run_drift_detection(reference_path: str = "data/customer_churn_real.csv") -> dict:
    os.makedirs("reports", exist_ok=True)

    if os.path.exists(reference_path):
        reference = pd.read_csv(reference_path)
    else:
        reference = generate_churn_dataset()

    current = simulate_production_traffic(base_path=reference_path)

    reference_features = reference[FEATURE_COLUMNS].copy()
    current_features = current[FEATURE_COLUMNS].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_features, current_data=current_features)

    report.save_html("reports/drift_report.html")
    result = report.as_dict()

    drift_metric = next(
        m for m in result["metrics"] if m["metric"] == "DatasetDriftMetric"
    )
    drift_by_column_metric = next(
        m for m in result["metrics"] if m["metric"] == "DataDriftTable"
    )

    n_drifted = drift_metric["result"]["number_of_drifted_columns"]
    n_columns = drift_metric["result"]["number_of_columns"]
    share_drifted = drift_metric["result"]["share_of_drifted_columns"]
    dataset_drift = drift_metric["result"]["dataset_drift"]

    drifted_features = [
        col
        for col, info in drift_by_column_metric["result"]["drift_by_columns"].items()
        if info["drift_detected"]
    ]

    if share_drifted >= RETRAIN_THRESHOLD_SHARE_DRIFTED:
        alert = True
        severity = "critical"
        recommended_action = (
            "TRIGGER_RETRAINING: majority of features have drifted from the "
            "training distribution. Schedule an immediate retrain on recent "
            "production data and re-validate before promoting a new model."
        )
    elif share_drifted >= ALERT_THRESHOLD_SHARE_DRIFTED:
        alert = True
        severity = "warning"
        recommended_action = (
            "INVESTIGATE_AND_MONITOR: meaningful drift detected in a subset "
            "of features. Increase monitoring frequency, validate upstream "
            "data sources, and plan a retrain if drift persists over the "
            "next monitoring window."
        )
    else:
        alert = False
        severity = "none"
        recommended_action = (
            "NO_ACTION: feature distributions are consistent with training "
            "data. Continue routine monitoring on the standard schedule."
        )

    summary = {
        "dataset_drift_detected": dataset_drift,
        "share_of_drifted_features": round(share_drifted, 4),
        "number_of_drifted_features": n_drifted,
        "total_features_monitored": n_columns,
        "drifted_features": drifted_features,
        "alert": alert,
        "severity": severity,
        "recommended_action": recommended_action,
        "reference_rows": len(reference_features),
        "current_rows": len(current_features),
        "reference_dataset": reference_path,
        "report_html": "reports/drift_report.html",
    }

    with open("reports/drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    summary = run_drift_detection()
    print(json.dumps(summary, indent=2))
    if summary["alert"]:
        print(f"\n[ALERT:{summary['severity'].upper()}] {summary['recommended_action']}")
        if summary["severity"] == "critical":
            sys.exit(1)  # non-zero exit lets CI gate on critical drift
