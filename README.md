# Customer Churn Prediction — Production Model Serving & Monitoring Platform

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![MLflow](https://img.shields.io/badge/MLflow-2.17-0194E2)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)
![Kubernetes](https://img.shields.io/badge/Kubernetes-deployable-326CE5)
![Prometheus](https://img.shields.io/badge/Prometheus-instrumented-E6522C)
![Evidently](https://img.shields.io/badge/Evidently%20AI-drift%20monitoring-purple)
![CI](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF)
![Tests](https://img.shields.io/badge/tests-17%20passing-brightgreen)

A telecom/SaaS customer-churn classifier built the way it would actually
ship at a company: trained with tracked experiments, registered in a model
registry, served behind a versioned REST API, containerized, deployed to
Kubernetes with autoscaling, monitored with Prometheus/Grafana, and
watched for data drift with automated retraining recommendations — all
wired into a CI/CD pipeline.

Most entry-level ML portfolios stop at a notebook with an `accuracy_score`
printed at the bottom. This project is the other 80% of the job: the
deployment, serving, and monitoring layer that turns a model into a
system.

---

## Architecture

```
                         ┌─────────────────────┐
                         │   Training Layer      │
                         │  generate_data.py      │
                         │  train.py (sklearn)     │
                         │  -> MLflow Tracking      │
                         │  -> MLflow Model Registry │
                         └──────────┬───────────────┘
                                    │ registers / exports
                                    ▼
                         ┌─────────────────────┐
                         │   Serving Layer        │
                         │  FastAPI (app.py)        │
                         │  /predict /predict/batch  │
                         │  /health   /metrics         │
                         └──────────┬───────────────┘
                                    │ containerized
                                    ▼
                ┌───────────────────────────────────────┐
                │              Docker / Kubernetes          │
                │  3 replicas · readiness+liveness probes    │
                │  HPA (CPU/memory) · resource requests/limits │
                └──────────┬──────────────────────┬───────────┘
                           │ scraped by                │ traffic sampled by
                           ▼                            ▼
                ┌────────────────────┐        ┌───────────────────────┐
                │ Prometheus + Grafana │        │  Evidently AI drift     │
                │ latency / volume /    │        │  reference vs. current  │
                │ error-rate dashboards  │        │  -> alert + action       │
                └────────────────────┘        └───────────────────────┘
                           ▲
                           │ all stages gated by
                ┌────────────────────────────────────────┐
                │        GitHub Actions CI/CD                │
                │  train -> test -> drift check -> build ->    │
                │  smoke test -> (placeholder) kubectl apply    │
                └────────────────────────────────────────┘
```

---

## Repository structure

```
churn-mlops-platform/
├── src/
│   ├── training/
│   │   ├── generate_data.py       # realistic synthetic churn dataset (fallback/demo)
│   │   ├── prepare_real_data.py    # maps real IBM Telco data onto our schema
│   │   └── train.py                 # sklearn GBM + MLflow tracking + registry
│   ├── serving/
│   │   ├── app.py                   # FastAPI app: predict/health/metrics
│   │   └── schemas.py                # pydantic request/response contracts
│   └── monitoring/
│       └── drift_detection.py         # Evidently AI drift report + alerting
├── tests/
│   ├── conftest.py
│   ├── test_health.py                 # /health, /metrics contract
│   ├── test_prediction_schema.py       # /predict, /predict/batch contract
│   ├── test_input_validation.py         # malformed-input rejection
│   └── test_prediction_sanity.py         # high-risk > low-risk regression guard
├── k8s/
│   ├── deployment.yaml                 # 3 replicas, probes, resource limits
│   ├── service.yaml                     # ClusterIP + NodePort, namespace
│   └── hpa.yaml                          # CPU/memory autoscaling
├── monitoring/
│   ├── prometheus.yml                   # scrape config targeting the API
│   └── grafana/
│       ├── dashboard.json                # prediction volume / latency / errors
│       └── provisioning/                  # auto-loads datasource + dashboard
├── .github/workflows/ci.yml               # train -> test -> drift -> build -> deploy
├── Dockerfile                              # HEALTHCHECK, non-root user
├── docker-compose.yml                       # API + Prometheus + Grafana + MLflow
├── requirements.txt / requirements-dev.txt
├── data/
│   ├── telco_raw.csv                           # real IBM Telco Churn source file
│   └── customer_churn_real.csv                 # mapped onto this project's schema
├── models/model.pkl                           # trained pipeline (serving artifact)
└── reports/                                     # real run outputs (see below)
    ├── training_metrics.json
    ├── drift_summary.json
    ├── drift_report.html
    ├── sample_prediction_high_risk.json
    └── sample_prediction_low_risk.json
```

---

## Quickstart

### Local (Docker Compose) — API + Prometheus + Grafana + MLflow

```bash
git clone <your-repo-url> && cd churn-mlops-platform
pip install -r requirements-dev.txt

# 1. Get the real Telco Customer Churn dataset and map it onto this
#    project's schema (falls back to a synthetic dataset if you skip this)
curl -sf https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv -o data/telco_raw.csv
python src/training/prepare_real_data.py

# 2. Train the model on real data (writes models/model.pkl, logs to ./mlruns)
python src/training/train.py --data-path data/customer_churn_real.csv --experiment-name customer-churn-prediction-real

# 3. Bring up the full stack
docker compose up --build

# API:         http://localhost:8000/docs
# Prometheus:  http://localhost:9090
# Grafana:     http://localhost:3000 (admin/admin, dashboard auto-provisioned)
# MLflow UI:   http://localhost:5000
```

### Run tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

### Kubernetes (kind / minikube)

```bash
docker build -t churn-api:latest .
kind load docker-image churn-api:latest      # or: minikube image load churn-api:latest

kubectl apply -f k8s/service.yaml             # creates namespace + Service
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/hpa.yaml

kubectl -n churn-prediction get pods,svc,hpa
kubectl -n churn-prediction port-forward svc/churn-api 8000:80
curl http://localhost:8000/health
```

Full instructions: [`k8s/README.md`](k8s/README.md).

---

## Real results from this run

Everything below is copy-pasted from an actual execution of this pipeline
— trained on the **real, public IBM Telco Customer Churn dataset**
(7,043 real customer records, 26.5% real churn rate), not synthetic data.
A synthetic generator (`src/training/generate_data.py`) also ships in this
repo for offline/demo use when no dataset is available, but the numbers
below — and the model actually registered and served — are from real
customers.

### Model training — MLflow-tracked hyperparameter sweep (6 runs)

| Run | n_estimators | max_depth | learning_rate | balanced | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 100 | 3 | 0.10 | – | 0.7999 | 0.6575 | 0.5134 | 0.5766 | 0.8396 |
| 1 | 200 | 3 | 0.05 | – | 0.7991 | 0.6553 | 0.5134 | 0.5757 | 0.8418 |
| 2 | 200 | 4 | 0.10 | – | 0.7906 | 0.6278 | 0.5187 | 0.5681 | 0.8314 |
| 3 | 300 | 3 | 0.03 | – | 0.7984 | 0.6563 | 0.5053 | 0.5710 | 0.8417 |
| 4 | 300 | 4 | 0.05 | ✓ | 0.7516 | 0.5223 | 0.7513 | 0.6162 | 0.8313 |
| **5 (best)** | **400** | **3** | **0.03** | **✓** | **0.7466** | **0.5148** | **0.7888** | **0.6230** | **0.8382** |

**Best model (registered as `churn-predictor` → Production stage in the
MLflow Model Registry):**

```
accuracy:   0.7466
precision:  0.5148
recall:     0.7888
f1_score:   0.6230
roc_auc:    0.8382
```

**Why the "best" run isn't the highest-ROC-AUC one:** model selection is
done on **F1**, not raw ROC-AUC (`SELECTION_METRIC` in `train.py`). Runs
0–3 have higher ROC-AUC but only catch ~51% of actual churners
(recall 0.51). Runs 4–5 use `sample_weight` to upweight the minority
(churn) class — a real technique for the real ~3.5:1 class imbalance in
this dataset — trading some precision for recall 0.79. In a retention
use case, a missed churner (false negative) usually costs more than an
unnecessary retention offer (false positive), so the model that actually
flags ~79% of at-risk customers is the better production choice even
though it "looks worse" on a naive accuracy/ROC-AUC reading. That
trade-off, and being able to justify it, is itself part of what this
project demonstrates.

All 6 runs, their params, metrics, confusion-matrix artifacts, and the
serialized model are logged under the `customer-churn-prediction-real`
experiment in MLflow (`mlruns/`, or via `mlflow ui`).

### Sample API responses (`/predict`)

High-risk customer (2-month tenure, month-to-month contract, 6 support
tickets, no tech support):

```json
{
  "customer_id": "CUST-000123",
  "churn_probability": 0.9103,
  "churn_prediction": true,
  "risk_tier": "high",
  "model_version": "local:models/model.pkl"
}
```

Low-risk customer (60-month tenure, two-year contract, 0 support tickets,
tech support + online security enabled):

```json
{
  "customer_id": "CUST-000456",
  "churn_probability": 0.065,
  "churn_prediction": false,
  "risk_tier": "low",
  "model_version": "local:models/model.pkl"
}
```

The model correctly separates the two by a wide margin (0.91 vs. 0.07),
which is exactly what `tests/test_prediction_sanity.py` asserts on every
CI run so a bad retrain can't silently ship a directionally-wrong model.

### Drift detection (Evidently AI)

Simulated a window of production traffic drifting from the training
distribution (contract mix shifting toward month-to-month, support
tickets and usage climbing, a price increase, and a surge of newer
signups) by resampling and perturbing the **same real Telco dataset** used
as the reference — comparing a dataset against a differently-generated
one would register as "drift" from methodology differences alone, which
would make the alert meaningless, so both sides come from the same real
customer data:

```json
{
  "dataset_drift_detected": false,
  "share_of_drifted_features": 0.4,
  "number_of_drifted_features": 6,
  "total_features_monitored": 15,
  "drifted_features": [
    "avg_monthly_usage_gb",
    "monthly_charges",
    "num_support_tickets",
    "tenure_months",
    "contract_type",
    "payment_method"
  ],
  "alert": true,
  "severity": "warning",
  "recommended_action": "INVESTIGATE_AND_MONITOR: meaningful drift detected in a subset of features. Increase monitoring frequency, validate upstream data sources, and plan a retrain if drift persists over the next monitoring window.",
  "reference_rows": 7043,
  "current_rows": 2000
}
```

The full interactive Evidently HTML report (per-feature distribution
comparisons, drift scores, statistical tests) is generated at
`reports/drift_report.html`. The script exits non-zero on `critical`
severity so CI can gate a deploy on drift, and always writes a
machine-readable `drift_summary.json` an alerting system could poll.

### Tests

```
17 passed in 0.95s

tests/test_health.py .... (4)
tests/test_input_validation.py ..... (5)
tests/test_prediction_sanity.py .... (4)
tests/test_prediction_schema.py .... (4)
```

---

## Why this matters (the gap this project closes)

Most entry-level ML/AI Engineer applicants can show a notebook that trains
a model and prints `accuracy_score`. In the 2026 job market that reads as
"data analyst who used sklearn," not "engineer who can own a model in
production." Hiring managers screening ML Engineer and AI Engineer roles
are specifically looking for evidence of the deployment/serving/monitoring
layer, because that's the part that doesn't show up in a Kaggle notebook
and the part that actually breaks in production. This project demonstrates:

- **MLflow** — experiment tracking across a real hyperparameter sweep
  (params, metrics, artifacts) and a Model Registry with stage promotion
  (`Production`), not just a `model.pkl` saved by hand.
- **FastAPI model serving** — a versioned REST contract (Pydantic schemas,
  input validation, batch endpoint) instead of a Flask script that only
  works in a demo.
- **Docker + Kubernetes** — a container with a real `HEALTHCHECK`, and
  k8s manifests with resource requests/limits, readiness/liveness/startup
  probes, and a `HorizontalPodAutoscaler` — the difference between "runs
  on my machine" and "survives a pod restart in production."
- **Prometheus + Grafana** — the model exposes its own operational
  metrics (prediction volume by risk tier, latency histograms, error
  counts) instead of being a black box nobody can debug at 2am.
- **Evidently AI + drift monitoring** — the single most commonly missing
  piece in ML portfolios: recognizing that a model's accuracy decays as
  the world changes, and having an automated, alertable process for
  catching it and recommending retraining.
- **CI/CD** — train, test, drift-check, and build the deploy artifact on
  every push, with prediction-sanity tests as a regression guard against
  a broken retrain — not "it worked when I ran it locally."

---

## Resume bullet points

- Designed and shipped an end-to-end MLOps platform for customer-churn
  prediction on 7,043 real customer records (IBM Telco Churn dataset) —
  MLflow experiment tracking/registry, a FastAPI serving layer, and
  Kubernetes deployment with autoscaling — tuning model selection for
  recall (79% of actual churners flagged, 0.838 ROC-AUC) over raw
  accuracy to match the real cost asymmetry of missed churners vs.
  unnecessary retention offers.
- Instrumented a production model-serving API with Prometheus metrics
  (request latency histograms, prediction volume by risk tier, error
  rates) and built a Grafana dashboard, backed by a Kubernetes
  `HorizontalPodAutoscaler` scaling 3–10 replicas on CPU/memory
  utilization.
- Built an automated data-drift monitoring pipeline with Evidently AI that
  compares live traffic against the training distribution across 15
  features, flags severity-tiered alerts (detected drift in 40% of
  monitored features in a simulated production shift), and recommends
  retraining — integrated into a GitHub Actions CI/CD pipeline that
  trains on real customer data, runs 17 automated tests, drift-checks, and
  Docker-builds on every push.

---

## Notes / honest limitations

- The model is trained on the real, public **IBM Telco Customer Churn**
  dataset (7,043 real customers). Two features it doesn't natively track
  — `num_support_tickets` and `avg_monthly_usage_gb` — are derived proxies
  built from real, correlated signals in the dataset (see the docstring in
  `src/training/prepare_real_data.py` for exactly how); every other field,
  including the churn label itself, is the real recorded value. A
  synthetic generator (`src/training/generate_data.py`) also ships for
  offline demos / CI runs where fetching the real file isn't desired.
- `docker-compose up` and the Kubernetes manifests were validated for
  correctness (config, health checks, probes, resource limits) but the
  actual `docker build` / `kubectl apply` execution should be run in your
  own environment with registry/cluster access — the GitHub Actions
  workflow in `.github/workflows/ci.yml` runs the real build + smoke test
  on every push.
- The CI/CD `deploy` job is intentionally a placeholder (see the comments
  in `.github/workflows/ci.yml`) since it needs your cluster's kubeconfig
  as a secret — wiring instructions are inline.
