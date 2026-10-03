from datetime import date as Date
from datetime import datetime
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.auth import Role
from app.domain import (
    FaultCause,
    InvestigationStatus,
    PredictionStatus,
    ResolutionStatus,
    VerificationStatus,
)


class KPIRecord(BaseModel):
    """The seven ordered model features plus optional history context."""

    model_config = ConfigDict(extra="forbid")

    retainability: float = Field(description="Retainability KPI used by the model.")
    hosr: float = Field(description="Handover success rate KPI used by the model.")
    rsrp: float = Field(description="RSRP KPI used by the model.")
    rsrq: float = Field(description="RSRQ KPI used by the model.")
    sinr: float = Field(description="SINR KPI used by the model.")
    average_throughput: float = Field(description="Average throughput KPI used by the model.")
    distance: float = Field(ge=0, description="Nonnegative distance KPI used by the model.")

    cell: str | None = Field(default=None, min_length=1, max_length=128)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    date: Date | None = None

    @field_validator(
        "retainability",
        "hosr",
        "rsrp",
        "rsrq",
        "sinr",
        "average_throughput",
        "distance",
        mode="before",
    )
    @classmethod
    def require_json_number(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("KPI values must be JSON numbers")
        return value

    @field_validator(
        "retainability", "hosr", "rsrp", "rsrq", "sinr", "average_throughput", "distance"
    )
    @classmethod
    def finite_values(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("KPI values must be finite numbers")
        return value


class PredictionRequest(KPIRecord):
    """Request for schema v1: seven KPI features; context is optional."""


class BinaryDetection(BaseModel):
    prediction: Literal["Fault", "Normal"] = Field(
        description="Binary model decision at threshold."
    )
    threshold: float = Field(description="Explicit configured fault threshold (0.5).")
    fault_probability: float = Field(ge=0, le=1)
    normal_probability: float = Field(ge=0, le=1)


class Classification(BaseModel):
    predicted_class_id: int = Field(ge=1, le=7)
    predicted_cause: FaultCause = Field(
        description=(
            "Raw multiclass winner: 1 ED, 2 CH, 3 II, 4 TLHO, 5 RP, 6 EU, 7 Normal. "
            "This is not a final cause when status is REVIEW_REQUIRED."
        )
    )
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float] = Field(
        description="Seven-class probability map keyed by ED, CH, II, TLHO, RP, EU, and Normal."
    )


class PredictionResponse(BaseModel):
    schema_version: Literal["v1"] = "v1"
    prediction_id: int
    status: PredictionStatus = Field(
        description="FAULT, NORMAL, or REVIEW_REQUIRED when binary and multiclass disagree."
    )
    binary_detection: BinaryDetection
    classification: Classification
    cause: FaultCause | None = None
    cause_name: str | None = None
    fault: bool | None = None
    recommendations: list[str]
    model_version: str
    created_at: datetime
    kpis: dict[str, object]


class OperationalPredictionResponse(BaseModel):
    """Prediction representation available to operational roles."""

    schema_version: Literal["v1"] = "v1"
    prediction_id: int
    status: PredictionStatus
    fault: bool | None = None
    cause: FaultCause | None = None
    cause_name: str | None = None
    recommendations: list[str]
    created_at: datetime
    kpis: dict[str, object]


class SiteHistoryItem(BaseModel):
    prediction_id: int
    cell: str | None
    latitude: float | None
    longitude: float | None
    fault: bool | None
    cause: FaultCause | None
    date: Date | None
    created_at: datetime
    status: PredictionStatus
    model_version: str
    kpis: dict[str, object]
    inference: dict[str, object]


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded", "not_ready"]
    database: Literal["ready", "unavailable"]
    inference: Literal["ready", "artifacts_unavailable"]


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=1024)


class AuthenticatedUser(BaseModel):
    id: int
    username: str
    is_active: bool
    role: Role


class InvestigationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: int | None = Field(default=None, ge=1)
    network_health_result_id: int | None = Field(default=None, ge=1)

    @field_validator("network_health_result_id", mode="after")
    @classmethod
    def require_one_source(cls, value: int | None, info) -> int | None:
        prediction_id = info.data.get("prediction_id")
        if (prediction_id is None) == (value is None):
            raise ValueError("Supply exactly one prediction_id or network_health_result_id.")
        return value


class InvestigationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findings: str | None = Field(default=None, max_length=10000)
    investigation_cause: FaultCause | None = None
    resolution_notes: str | None = Field(default=None, max_length=10000)
    is_completed: bool | None = None


class InvestigationVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verification_status: VerificationStatus
    verified_cause: FaultCause | None = None
    notes: str = Field(default="", max_length=5000)


class InvestigationOutcome(BaseModel):
    investigation_id: int
    prediction_id: int | None
    network_health_result_id: int | None
    cell: str | None
    predicted_cause: FaultCause | None
    investigation_status: InvestigationStatus
    findings: str
    investigation_cause: FaultCause | None
    resolution_notes: str
    submitted_by: int
    submitted_by_name: str
    created_at: datetime
    updated_at: datetime
    submitted_at: datetime | None
    verified_cause: FaultCause | None
    verified_by: int | None
    verified_by_name: str | None
    verified_at: datetime | None
    verification_status: VerificationStatus
    resolution_status: ResolutionStatus
    is_completed: bool
    evaluation_eligible: bool
    verification_history: list[dict[str, object]]
    operational_context: dict[str, object] | None = None
