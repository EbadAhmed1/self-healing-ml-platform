"""
app/schemas.py
──────────────
Pydantic request/response models for the churn prediction endpoint.

VALIDATION PHILOSOPHY:
  Every field is explicitly typed and constrained. FastAPI will return a 422
  Unprocessable Entity (with a structured error body) on ANY validation
  failure BEFORE the request reaches the model — bad input never silently
  flows through to the pipeline.

  String fields are stripped and lowercased where the raw data has a known
  canonical form, so callers aren't penalised for case inconsistencies.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class ChurnInput(BaseModel):
    """Input features for a single churn prediction request.

    Field names and allowed values mirror the Telco dataset exactly so that
    model/feature_config.py and this schema stay in sync.
    """

    # ── Numeric features ────────────────────────────────────────────────────
    tenure: float = Field(
        ..., ge=0, description="Months the customer has been with the company."
    )
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charge in USD.")
    TotalCharges: float = Field(..., ge=0, description="Total charges to date in USD.")

    # ── Categorical features ─────────────────────────────────────────────────
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender.")
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description="1 if senior citizen, 0 otherwise."
    )
    Partner: Literal["Yes", "No"] = Field(..., description="Has a partner.")
    Dependents: Literal["Yes", "No"] = Field(..., description="Has dependents.")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Has phone service.")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Has multiple phone lines."
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Internet service provider type."
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Has online security add-on."
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Has online backup add-on."
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Has device protection add-on."
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Has tech support add-on."
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Has streaming TV add-on."
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Has streaming movies add-on."
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Contract term."
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Uses paperless billing."
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Payment method.")

    @model_validator(mode="after")
    def total_charges_consistency(self) -> "ChurnInput":
        """
        Soft sanity check: TotalCharges should not exceed tenure * MonthlyCharges
        by more than 10% (rounding + promotions are allowed).
        This catches obvious data-entry errors without being too strict.
        """
        if self.tenure > 0:
            expected_max = self.tenure * self.MonthlyCharges * 1.10
            if self.TotalCharges > expected_max + 1:  # +1 to avoid float noise
                raise ValueError(
                    f"TotalCharges ({self.TotalCharges}) is implausibly high "
                    f"given tenure={self.tenure} months and "
                    f"MonthlyCharges={self.MonthlyCharges} "
                    f"(expected ≤ {expected_max:.2f}). Check your input."
                )
        return self

    def to_dataframe(self):
        """Convert to a single-row DataFrame matching the training feature order."""
        import pandas as pd
        from models.feature_config import ALL_FEATURES

        row = {k: v for k, v in self.model_dump().items() if k in set(ALL_FEATURES)}
        return pd.DataFrame([row])[ALL_FEATURES]

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 24,
                "MonthlyCharges": 65.5,
                "TotalCharges": 1571.0,
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Bank transfer (automatic)",
            }
        }
    }


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
class ChurnPrediction(BaseModel):
    """Response returned by POST /predict/churn-model."""

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "prediction": False,
                "prediction_label": "No Churn",
                "confidence": 0.812,
                "model_id": "churn-model:v1",
            }
        },
    }

    prediction: bool = Field(..., description="True = churn predicted.")
    prediction_label: str = Field(
        ..., description="Human-readable label: 'Churn' or 'No Churn'."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of the predicted class."
    )
    model_id: str = Field(..., description="Model identifier used for this prediction.")
    prediction_id: str | None = Field(
        default=None,
        description="Database UUID primary key for this prediction record.",
    )


# ---------------------------------------------------------------------------
# Tenant #2: Fraud Detection Schemas
# ---------------------------------------------------------------------------
class FraudInput(BaseModel):
    """Input features for single transaction fraud prediction."""

    transaction_amount: float = Field(
        ..., ge=0, description="Transaction amount in USD."
    )
    location_distance_km: float = Field(
        ..., ge=0, description="Distance from cardholder home in km."
    )
    account_age_days: float = Field(..., ge=0, description="Account age in days.")
    velocity_1h_count: float = Field(
        ..., ge=0, description="Transactions count in last 1 hour."
    )

    merchant_category: Literal[
        "retail", "travel", "gaming", "crypto", "electronics"
    ] = Field(..., description="Merchant category.")
    device_type: Literal["mobile_ios", "mobile_android", "web_desktop"] = Field(
        ..., description="Device operating system."
    )
    is_international: Literal[0, 1] = Field(
        ..., description="1 if international transaction, 0 otherwise."
    )
    is_flagged_ip: Literal[0, 1] = Field(
        ..., description="1 if IP is flagged on blocklist, 0 otherwise."
    )

    def to_dataframe(self):
        import pandas as pd
        from models.feature_config_fraud import ALL_FEATURES

        row = self.model_dump()
        return pd.DataFrame([row])[ALL_FEATURES]


class FraudPrediction(BaseModel):
    """Response returned by POST /predict/fraud-model."""

    model_config = {"protected_namespaces": ()}

    prediction: bool = Field(..., description="True = fraud predicted.")
    prediction_label: str = Field(..., description="'Fraud' or 'Legitimate'.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score.")
    model_id: str = Field(..., description="Model identifier used for this prediction.")


# ---------------------------------------------------------------------------
# Health response schema
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    model_id: str | None = None
