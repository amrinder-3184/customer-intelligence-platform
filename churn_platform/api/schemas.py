"""Pydantic request/response models for the churn API.

The request mirrors the 19 model predictors EXACTLY (same names as the CSV
columns), with the category values found in the real dataset (see
validation.CATEGORICAL_COLUMNS). A test checks that these Literal values,
the Phase 1 validator and the fitted one-hot encoder all agree, so the
schema cannot silently drift away from what the model was trained on.

Validation happens before any model code runs:
- missing field         -> 422
- unknown category      -> 422 (e.g. Contract="Three year")
- wrong type / NaN / inf / out-of-range number -> 422
- extra field, including customerID -> 422 (extra="forbid")
- impossible combinations the training data never contains -> 422
"""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

YesNo = Literal["Yes", "No"]
InternetAddOn = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    """One customer's attributes. customerID is deliberately NOT accepted."""

    model_config = ConfigDict(
        extra="forbid",          # unknown keys are errors, never silently ignored or used
        str_strip_whitespace=False,
        json_schema_extra={"example": {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.5, "TotalCharges": 430.25,
        }},
    )

    # Numeric. strict=True rejects strings like "12" and booleans; allow_inf_nan=False rejects inf/NaN.
    tenure: int = Field(ge=0, le=120, strict=True, description="Months as a customer")
    MonthlyCharges: float = Field(ge=0, le=1000, allow_inf_nan=False, strict=True)
    TotalCharges: float | None = Field(
        ge=0, le=100_000, allow_inf_nan=False, strict=True,
        description="Required key. null is allowed only for tenure 0 (not yet billed), as in the training data.")

    # Categorical
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(description="0 = no, 1 = yes (as in the dataset)")
    Partner: YesNo
    Dependents: YesNo
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddOn
    OnlineBackup: InternetAddOn
    DeviceProtection: InternetAddOn
    TechSupport: InternetAddOn
    StreamingTV: InternetAddOn
    StreamingMovies: InternetAddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]

    @field_validator("SeniorCitizen", mode="before")
    @classmethod
    def reject_bool_senior(cls, v):
        # In Python, True == 1, so Literal[0, 1] would otherwise accept true/false.
        if isinstance(v, bool):
            raise ValueError("SeniorCitizen must be the integer 0 or 1, not a boolean")
        return v

    @model_validator(mode="after")
    def consistent_with_training_data(self):
        """The same cross-column rules as the Phase 1 validator; training data never breaks them."""
        if (self.PhoneService == "No") != (self.MultipleLines == "No phone service"):
            raise ValueError("MultipleLines must be 'No phone service' exactly when PhoneService is 'No'")
        no_internet = self.InternetService == "No"
        for name in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                     "StreamingTV", "StreamingMovies"):
            if no_internet != (getattr(self, name) == "No internet service"):
                raise ValueError(f"{name} must be 'No internet service' exactly when InternetService is 'No'")
        if self.TotalCharges is None and self.tenure != 0:
            raise ValueError("TotalCharges may be null only when tenure is 0 (customer not yet billed)")
        return self


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1, description="Model-estimated probability of churn")
    predicted_churn: bool = Field(description="churn_probability >= threshold")
    risk_label: Literal["low", "elevated", "high"]
    threshold: float
    model_name: str
    model_version: str
    note: str = ("Model estimate from a retrospective model trained on historical data; "
                 "not a guarantee that this customer will or will not churn.")

    @field_validator("churn_probability")
    @classmethod
    def finite(cls, v):
        if not math.isfinite(v):
            raise ValueError("probability must be finite")
        return v


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    created_at_utc: str
    n_predictors: int
    predictors: list[dict]
    target: str
    threshold: float
    threshold_method: str
    cost_assumptions: dict
    risk_bands: dict
    training_summary: dict
    historical_evaluation: dict
    limitations: list[str]
