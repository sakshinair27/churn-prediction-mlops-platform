"""
Maps the public IBM Telco Customer Churn dataset (7,043 real customer
records) onto this project's feature schema, so the model is trained and
evaluated on real-world churn data rather than only the synthetic
generator in generate_data.py.

Source: IBM Telco Customer Churn sample dataset (widely used public
benchmark for churn modeling), columns: customerID, gender, SeniorCitizen,
Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService,
OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges,
TotalCharges, Churn.

We do not have real "support ticket count" or "usage GB" fields in this
public dataset (most public Telco churn datasets don't track that), so
those two columns are derived as reasonable proxies (documented below)
rather than fabricated outright -- everything else is the real recorded
value for a real customer.
"""
import numpy as np
import pandas as pd

RAW_PATH = "data/telco_raw.csv"
OUT_PATH = "data/customer_churn_real.csv"


def prepare(raw_path: str = RAW_PATH, out_path: str = OUT_PATH, seed: int = 7) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    # TotalCharges has some blank strings for brand-new customers (tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    rng = np.random.default_rng(seed)
    n = len(df)

    has_tech_support = (df["TechSupport"] == "Yes").astype(int)
    has_online_security = (df["OnlineSecurity"] == "Yes").astype(int)

    # Proxy for support-ticket volume: this public dataset does not track
    # ticket counts, so we derive a plausible count from real signals that
    # correlate with support burden in the actual data (no tech support,
    # no online security, fiber internet, short tenure), plus noise. This
    # keeps every OTHER field (contract, tenure, charges, churn label)
    # as the real recorded value.
    ticket_signal = (
        (1 - has_tech_support) * 1.1
        + (1 - has_online_security) * 0.9
        + (df["InternetService"] == "Fiber optic").astype(int) * 0.6
        + (df["tenure"] < 6).astype(int) * 1.0
    )
    num_support_tickets = rng.poisson(lam=np.clip(ticket_signal, 0.1, None))

    # Proxy for usage volume: derived from MonthlyCharges + service mix
    # (fiber and streaming add-ons imply heavier usage), again because raw
    # bandwidth usage isn't in the public dataset.
    streaming = ((df["StreamingTV"] == "Yes").astype(int) + (df["StreamingMovies"] == "Yes").astype(int))
    base_usage = df["MonthlyCharges"] * 1.8 + streaming * 40
    avg_monthly_usage_gb = (base_usage * rng.normal(1.0, 0.15, n)).clip(5, 1000).round(1)

    additional_service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    num_additional_services = (df[additional_service_cols] == "Yes").sum(axis=1)

    out = pd.DataFrame(
        {
            "customer_id": df["customerID"],
            "tenure_months": df["tenure"].astype(int),
            "contract_type": df["Contract"],
            "monthly_charges": df["MonthlyCharges"].astype(float),
            "total_charges": df["TotalCharges"].astype(float),
            "num_support_tickets": num_support_tickets,
            "has_tech_support": has_tech_support,
            "has_online_security": has_online_security,
            "internet_service": df["InternetService"],
            "payment_method": df["PaymentMethod"],
            "paperless_billing": (df["PaperlessBilling"] == "Yes").astype(int),
            "senior_citizen": df["SeniorCitizen"].astype(int),
            "partner": (df["Partner"] == "Yes").astype(int),
            "dependents": (df["Dependents"] == "Yes").astype(int),
            "num_additional_services": num_additional_services,
            "avg_monthly_usage_gb": avg_monthly_usage_gb,
            "churn": (df["Churn"] == "Yes").astype(int),
        }
    )

    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    out = prepare()
    print(f"Prepared {len(out)} real customer records -> {OUT_PATH}")
    print(f"Real churn rate: {out['churn'].mean():.4f}")
    print(out.head(3).T)
