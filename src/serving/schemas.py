"""Pydantic request/response schemas for the churn prediction API."""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_TYPES = ("Month-to-month", "One year", "Two year")
INTERNET_SERVICES = ("DSL", "Fiber optic", "No")
PAYMENT_METHODS = (
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
)


class CustomerFeatures(BaseModel):
    customer_id: Optional[str] = Field(default=None, description="Optional customer identifier")
    tenure_months: int = Field(..., ge=0, le=100, description="Months as a customer")
    contract_type: Literal[CONTRACT_TYPES]
    monthly_charges: float = Field(..., ge=0, le=1000)
    total_charges: float = Field(..., ge=0, le=100000)
    num_support_tickets: int = Field(..., ge=0, le=100)
    has_tech_support: int = Field(..., ge=0, le=1)
    has_online_security: int = Field(..., ge=0, le=1)
    internet_service: Literal[INTERNET_SERVICES]
    payment_method: Literal[PAYMENT_METHODS]
    paperless_billing: int = Field(..., ge=0, le=1)
    senior_citizen: int = Field(..., ge=0, le=1)
    partner: int = Field(..., ge=0, le=1)
    dependents: int = Field(..., ge=0, le=1)
    num_additional_services: int = Field(..., ge=0, le=20)
    avg_monthly_usage_gb: float = Field(..., ge=0, le=10000)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "CUST-000123",
                "tenure_months": 2,
                "contract_type": "Month-to-month",
                "monthly_charges": 95.50,
                "total_charges": 191.00,
                "num_support_tickets": 5,
                "has_tech_support": 0,
                "has_online_security": 0,
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check",
                "paperless_billing": 1,
                "senior_citizen": 0,
                "partner": 0,
                "dependents": 0,
                "num_additional_services": 0,
                "avg_monthly_usage_gb": 210.5,
            }
        }
    )


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    customer_id: Optional[str]
    churn_probability: float
    churn_prediction: bool
    risk_tier: Literal["low", "medium", "high"]
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: str
