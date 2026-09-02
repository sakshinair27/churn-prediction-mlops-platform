"""
FastAPI serving layer for the customer-churn model.

Endpoints:
  POST /predict         - single-customer prediction
  POST /predict/batch    - batch prediction (up to 1000 customers)
  GET  /health            - liveness/readiness probe for k8s
  GET  /metrics            - Prometheus exposition-format metrics

The model is loaded once at startup either from the MLflow Model Registry
(if MLFLOW_TRACKING_URI + MODEL_STAGE resolve) or, as a robust fallback for
local/offline serving, from the models/model.pkl artifact produced by
src/training/train.py.
"""
import logging
import os
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-serving")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")
MODEL_NAME = os.environ.get("MODEL_NAME", "churn-predictor")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

HIGH_RISK_THRESHOLD = 0.6
MEDIUM_RISK_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
PREDICTION_COUNTER = Counter(
    "churn_predictions_total",
    "Total number of churn predictions served, labeled by risk tier",
    ["risk_tier"],
)
REQUEST_LATENCY = Histogram(
    "churn_request_latency_seconds",
    "Request latency in seconds for prediction endpoints",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
ERROR_COUNTER = Counter(
    "churn_prediction_errors_total",
    "Total number of prediction request errors",
    ["endpoint", "error_type"],
)
REQUEST_COUNTER = Counter(
    "churn_requests_total",
    "Total requests received, labeled by endpoint and status",
    ["endpoint", "status_code"],
)

MODEL_STATE = {"pipeline": None, "version": "unknown", "loaded": False}


def _risk_tier(prob: float) -> str:
    if prob >= HIGH_RISK_THRESHOLD:
        return "high"
    if prob >= MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


def load_model():
    """Load the model, preferring the MLflow Model Registry, falling back to
    the local joblib artifact so the service still starts in environments
    without a reachable MLflow tracking server (e.g. a bare docker run)."""
    if MLFLOW_TRACKING_URI:
        try:
            import mlflow.pyfunc

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            pipeline = mlflow.pyfunc.load_model(model_uri)
            MODEL_STATE["pipeline"] = pipeline
            MODEL_STATE["version"] = f"mlflow:{MODEL_NAME}/{MODEL_STAGE}"
            MODEL_STATE["loaded"] = True
            logger.info("Loaded model from MLflow registry: %s", model_uri)
            return
        except Exception as e:
            logger.warning("Falling back to local model artifact: %s", e)

    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        MODEL_STATE["pipeline"] = pipeline
        MODEL_STATE["version"] = f"local:{MODEL_PATH}"
        MODEL_STATE["loaded"] = True
        logger.info("Loaded model from local artifact: %s", MODEL_PATH)
    else:
        logger.error("No model artifact found at %s", MODEL_PATH)
        MODEL_STATE["loaded"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production model-serving API for the customer churn classifier.",
    version="1.0.0",
    lifespan=lifespan,
)


def _to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    row = customer.model_dump(exclude={"customer_id"})
    return pd.DataFrame([row])


def _predict_one(customer: CustomerFeatures) -> PredictionResponse:
    df = _to_dataframe(customer)
    pipeline = MODEL_STATE["pipeline"]
    try:
        proba = float(pipeline.predict_proba(df)[:, 1][0])
    except AttributeError:
        # mlflow.pyfunc models expose predict() only; use it directly.
        proba = float(pipeline.predict(df)[0])
    tier = _risk_tier(proba)
    PREDICTION_COUNTER.labels(risk_tier=tier).inc()
    return PredictionResponse(
        customer_id=customer.customer_id,
        churn_probability=round(proba, 4),
        churn_prediction=proba >= 0.5,
        risk_tier=tier,
        model_version=MODEL_STATE["version"],
    )


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    endpoint = request.url.path
    try:
        response = await call_next(request)
    except Exception as e:
        ERROR_COUNTER.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
        REQUEST_COUNTER.labels(endpoint=endpoint, status_code="500").inc()
        raise
    duration = time.perf_counter() - start
    if endpoint in ("/predict", "/predict/batch"):
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    REQUEST_COUNTER.labels(endpoint=endpoint, status_code=str(response.status_code)).inc()
    if response.status_code >= 400:
        ERROR_COUNTER.labels(endpoint=endpoint, error_type=f"http_{response.status_code}").inc()
    return response


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """Liveness/readiness probe. Returns 200 only when the model is loaded."""
    if not MODEL_STATE["loaded"]:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "model_loaded": False, "model_version": "none"},
        )
    return HealthResponse(status="healthy", model_loaded=True, model_version=MODEL_STATE["version"])


@app.get("/metrics", tags=["ops"])
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(customer: CustomerFeatures):
    if not MODEL_STATE["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return _predict_one(customer)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
async def predict_batch(request: BatchPredictionRequest):
    if not MODEL_STATE["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        predictions = [_predict_one(c) for c in request.customers]
        return BatchPredictionResponse(predictions=predictions, count=len(predictions))
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


@app.get("/", tags=["ops"])
async def root():
    return {
        "service": "Customer Churn Prediction API",
        "status": "running",
        "model_loaded": MODEL_STATE["loaded"],
        "docs": "/docs",
    }
