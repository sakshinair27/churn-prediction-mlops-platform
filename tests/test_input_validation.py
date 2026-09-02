import copy

from .conftest import HIGH_RISK_CUSTOMER


def test_rejects_negative_tenure(client):
    payload = copy.deepcopy(HIGH_RISK_CUSTOMER)
    payload["tenure_months"] = -5
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_rejects_invalid_contract_type(client):
    payload = copy.deepcopy(HIGH_RISK_CUSTOMER)
    payload["contract_type"] = "Lifetime"
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_rejects_out_of_range_binary_flag(client):
    payload = copy.deepcopy(HIGH_RISK_CUSTOMER)
    payload["has_tech_support"] = 3
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_rejects_missing_required_field(client):
    payload = copy.deepcopy(HIGH_RISK_CUSTOMER)
    del payload["monthly_charges"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_rejects_negative_charges(client):
    payload = copy.deepcopy(HIGH_RISK_CUSTOMER)
    payload["monthly_charges"] = -10.0
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
