import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.domain import CAUSE_NAMES, FaultCause, PredictionStatus
from app.schemas import KPIRecord

logger = logging.getLogger(__name__)
CLASS_CODES = ("ED", "CH", "II", "TLHO", "RP", "EU", "Normal")


class InferenceUnavailable(RuntimeError):
    """Raised when local model or preprocessing artifacts cannot be loaded."""


class InferenceFailed(RuntimeError):
    """Raised when loaded artifacts fail while processing a request."""


@dataclass(frozen=True)
class InferenceResult:
    status: PredictionStatus
    fault: bool | None
    cause: str | None
    cause_name: str | None
    binary_prediction: str
    fault_probability: float
    normal_probability: float
    threshold: float
    class_id: int
    predicted_cause: str
    confidence: float
    class_probabilities: dict[str, float]
    model_version: str


class InferenceEngine(Protocol):
    @property
    def available(self) -> bool: ...

    def predict(self, record: KPIRecord) -> InferenceResult: ...


def resolve_status(binary_prediction: str, multiclass_class_id: int) -> PredictionStatus:
    """Resolve model agreement without suppressing either model's raw output."""
    binary_fault = binary_prediction == "Fault"
    multiclass_fault = multiclass_class_id in range(1, 7)
    if binary_fault and multiclass_fault:
        return PredictionStatus.FAULT
    if not binary_fault and not multiclass_fault:
        return PredictionStatus.NORMAL
    return PredictionStatus.REVIEW_REQUIRED


class ArtifactInference:
    """Load the modern Keras models/scaler once and adapt their output for the API."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self._engine = None
        self._load_error: Exception | None = None
        try:
            # Import TensorFlow only when the API attempts to load installed artifacts.
            from app.ml.inference import ModelInference

            self._engine = ModelInference(self.model_dir)
        except Exception as exc:
            self._load_error = exc
            logger.exception("ml_artifacts_load_failed")

    @property
    def available(self) -> bool:
        return self._engine is not None

    def predict(self, record: KPIRecord) -> InferenceResult:
        if self._engine is None:
            raise InferenceUnavailable(
                "Validated inference artifacts are unavailable. Check server configuration."
            ) from self._load_error

        return self.predict_batch([record])[0]

    def predict_batch(self, records: list[object]) -> list[InferenceResult]:
        if self._engine is None:
            raise InferenceUnavailable(
                "Validated inference artifacts are unavailable. Check server configuration."
            ) from self._load_error
        from app.ml.data import FEATURE_ORDER

        matrix = []
        for record in records:
            matrix.append(
                [
                    getattr(record, name.lower() if name != "Throughput" else "average_throughput")
                    for name in FEATURE_ORDER
                ]
            )
        try:
            raw_results = self._engine.predict_batch(matrix)
        except Exception as exc:
            logger.exception("ml_inference_failed")
            raise InferenceFailed("The inference service could not process this request.") from exc

        results = []
        for raw in raw_results:
            results.append(self._adapt(raw))
        return results

    @staticmethod
    def _adapt(raw: dict[str, object]) -> InferenceResult:

        binary_prediction = str(raw["binary_prediction"])
        class_id = int(raw["class_id"])
        predicted_cause = str(raw["predicted_cause"])
        cause_enum = FaultCause(predicted_cause)
        status = resolve_status(binary_prediction, class_id)
        cause = (
            cause_enum.value
            if status in (PredictionStatus.FAULT, PredictionStatus.NORMAL)
            else None
        )
        is_fault = (
            True
            if status == PredictionStatus.FAULT
            else False
            if status == PredictionStatus.NORMAL
            else None
        )
        return InferenceResult(
            status=status,
            fault=is_fault,
            cause=cause,
            cause_name=CAUSE_NAMES[cause_enum] if cause is not None else None,
            binary_prediction=binary_prediction,
            fault_probability=float(raw["fault_probability"]),
            normal_probability=1.0 - float(raw["fault_probability"]),
            threshold=float(raw["threshold"]),
            class_id=class_id,
            predicted_cause=predicted_cause,
            confidence=float(raw["confidence"]),
            class_probabilities={
                code: float(raw["class_probabilities"][str(index)])
                for index, code in enumerate(CLASS_CODES, start=1)
            },
            model_version=str(raw["model_version"]),
        )
