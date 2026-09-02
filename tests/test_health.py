def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_version"] != "none"


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"


def test_metrics_endpoint_exposes_prometheus_format(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    body = resp.text
    assert "churn_predictions_total" in body
    assert "churn_request_latency_seconds" in body
    assert "churn_prediction_errors_total" in body
