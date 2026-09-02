"""Model sanity checks: the classifier must rank obviously risky customers
above obviously loyal ones -- a core regression guard for CI so a bad
retrain can't silently ship a model that's directionally wrong."""
from .conftest import HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER


def test_high_risk_customer_scores_higher_than_low_risk(client):
    high = client.post("/predict", json=HIGH_RISK_CUSTOMER).json()
    low = client.post("/predict", json=LOW_RISK_CUSTOMER).json()
    assert high["churn_probability"] > low["churn_probability"]


def test_high_risk_customer_flagged_high_or_medium_tier(client):
    high = client.post("/predict", json=HIGH_RISK_CUSTOMER).json()
    assert high["risk_tier"] in {"high", "medium"}


def test_low_risk_customer_flagged_low_or_medium_tier(client):
    low = client.post("/predict", json=LOW_RISK_CUSTOMER).json()
    assert low["risk_tier"] in {"low", "medium"}


def test_month_to_month_increases_churn_risk_vs_two_year(client):
    """Holding everything else fixed, a month-to-month contract should not
    look safer than a two-year contract -- a monotonicity sanity check on
    the single strongest churn driver in telecom data."""
    base = dict(LOW_RISK_CUSTOMER)
    two_year = dict(base, contract_type="Two year")
    month_to_month = dict(base, contract_type="Month-to-month")

    p_two_year = client.post("/predict", json=two_year).json()["churn_probability"]
    p_mtm = client.post("/predict", json=month_to_month).json()["churn_probability"]

    assert p_mtm >= p_two_year
