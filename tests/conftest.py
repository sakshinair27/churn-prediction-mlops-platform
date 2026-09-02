import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from fastapi.testclient import TestClient

from serving.app import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


HIGH_RISK_CUSTOMER = {
    "customer_id": "TEST-HIGH",
    "tenure_months": 1,
    "contract_type": "Month-to-month",
    "monthly_charges": 105.0,
    "total_charges": 105.0,
    "num_support_tickets": 7,
    "has_tech_support": 0,
    "has_online_security": 0,
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "paperless_billing": 1,
    "senior_citizen": 1,
    "partner": 0,
    "dependents": 0,
    "num_additional_services": 0,
    "avg_monthly_usage_gb": 350.0,
}

LOW_RISK_CUSTOMER = {
    "customer_id": "TEST-LOW",
    "tenure_months": 65,
    "contract_type": "Two year",
    "monthly_charges": 40.0,
    "total_charges": 2600.0,
    "num_support_tickets": 0,
    "has_tech_support": 1,
    "has_online_security": 1,
    "internet_service": "DSL",
    "payment_method": "Credit card (automatic)",
    "paperless_billing": 0,
    "senior_citizen": 0,
    "partner": 1,
    "dependents": 1,
    "num_additional_services": 5,
    "avg_monthly_usage_gb": 60.0,
}
