from .conftest import HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER


def test_predict_response_schema(client):
    resp = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "customer_id",
        "churn_probability",
        "churn_prediction",
        "risk_tier",
        "model_version",
    }
    assert isinstance(body["churn_probability"], float)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)
    assert body["risk_tier"] in {"low", "medium", "high"}


def test_batch_predict_response_schema(client):
    resp = client.post(
        "/predict/batch", json={"customers": [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    for pred in body["predictions"]:
        assert 0.0 <= pred["churn_probability"] <= 1.0


def test_batch_predict_respects_max_size(client):
    customers = [HIGH_RISK_CUSTOMER] * 1001
    resp = client.post("/predict/batch", json={"customers": customers})
    assert resp.status_code == 422


def test_batch_predict_rejects_empty_list(client):
    resp = client.post("/predict/batch", json={"customers": []})
    assert resp.status_code == 422
