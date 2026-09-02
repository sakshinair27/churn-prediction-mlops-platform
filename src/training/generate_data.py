"""
Synthetic-but-realistic telecom/SaaS customer churn dataset generator.

Produces a dataset with the same shape and behavior as the classic
"Telco Customer Churn" style datasets used in industry: tenure, contract
type, monthly/total charges, service usage, and support-ticket history,
with churn probability driven by a believable underlying signal (short
tenure + month-to-month contract + high support tickets + high charges
=> higher churn risk) plus noise, so the resulting classifier has to do
real work instead of memorizing a trivial rule.
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 8000

CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
INTERNET_SERVICE = ["DSL", "Fiber optic", "No"]
PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


def generate_churn_dataset(n_customers: int = N_CUSTOMERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure_months = rng.gamma(shape=2.0, scale=18, size=n_customers).clip(0, 72).round().astype(int)

    contract_type = rng.choice(
        CONTRACT_TYPES, size=n_customers, p=[0.55, 0.25, 0.20]
    )
    internet_service = rng.choice(
        INTERNET_SERVICE, size=n_customers, p=[0.35, 0.45, 0.20]
    )
    payment_method = rng.choice(PAYMENT_METHODS, size=n_customers, p=[0.30, 0.20, 0.25, 0.25])

    base_monthly = rng.normal(65, 25, size=n_customers)
    fiber_bump = np.where(internet_service == "Fiber optic", 25, 0)
    monthly_charges = (base_monthly + fiber_bump).clip(15, 150).round(2)

    total_charges = (monthly_charges * tenure_months * rng.normal(1.0, 0.05, size=n_customers)).clip(0)
    total_charges = total_charges.round(2)

    num_support_tickets = rng.poisson(lam=1.2, size=n_customers)
    num_support_tickets += np.where(rng.random(n_customers) < 0.1, rng.integers(3, 8, n_customers), 0)

    has_tech_support = rng.choice([0, 1], size=n_customers, p=[0.55, 0.45])
    has_online_security = rng.choice([0, 1], size=n_customers, p=[0.6, 0.4])
    paperless_billing = rng.choice([0, 1], size=n_customers, p=[0.4, 0.6])
    senior_citizen = rng.choice([0, 1], size=n_customers, p=[0.84, 0.16])
    partner = rng.choice([0, 1], size=n_customers, p=[0.52, 0.48])
    dependents = rng.choice([0, 1], size=n_customers, p=[0.7, 0.3])
    num_additional_services = rng.integers(0, 6, size=n_customers)
    avg_monthly_usage_gb = rng.gamma(shape=3.0, scale=40, size=n_customers).clip(0, 800).round(1)

    # ---- Latent churn "risk score" (drives the label, not directly observed) ----
    risk = np.zeros(n_customers)
    risk += np.where(contract_type == "Month-to-month", 1.6, 0.0)
    risk += np.where(contract_type == "One year", 0.4, 0.0)
    risk += (24 - np.clip(tenure_months, 0, 24)) / 24 * 1.8
    risk += (num_support_tickets.clip(0, 10)) * 0.28
    risk += np.where(has_tech_support == 0, 0.5, -0.2)
    risk += np.where(has_online_security == 0, 0.35, -0.15)
    risk += np.where(payment_method == "Electronic check", 0.55, 0.0)
    risk += (monthly_charges - monthly_charges.mean()) / monthly_charges.std() * 0.45
    risk += np.where(senior_citizen == 1, 0.25, 0.0)
    risk += np.where(partner == 1, -0.3, 0.0)
    risk += np.where(dependents == 1, -0.25, 0.0)
    risk += np.where(paperless_billing == 1, 0.15, 0.0)
    risk -= num_additional_services * 0.12
    risk += rng.normal(0, 0.9, size=n_customers)  # noise so it's not trivially separable

    # Shift so the base churn rate lands near the ~26-27% industry-typical
    # telecom churn rate rather than an artificially balanced 50/50 split.
    churn_prob = 1 / (1 + np.exp(-(risk - risk.mean() - 0.95)))
    churn = (rng.random(n_customers) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(n_customers)],
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_tickets": num_support_tickets,
            "has_tech_support": has_tech_support,
            "has_online_security": has_online_security,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "paperless_billing": paperless_billing,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "num_additional_services": num_additional_services,
            "avg_monthly_usage_gb": avg_monthly_usage_gb,
            "churn": churn,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_churn_dataset()
    out_path = "data/customer_churn.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(f"Churn rate: {df['churn'].mean():.4f}")
    print(df.describe(include='all').T)
