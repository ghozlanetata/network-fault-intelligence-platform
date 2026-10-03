from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from app.ml.data import CLASS_MAPPING, FEATURE_ORDER

BINARY_FAULT_THRESHOLD = 0.5


class InferenceValidationError(ValueError):
    def __init__(self, code: str, message: str, details: dict[str, object] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def as_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": str(self), "details": self.details}


class ModelArtifactError(RuntimeError):
    pass


class ModelInference:
    """Loads saved artifacts once and transforms inputs without fitting preprocessing."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        preprocessing_path = self.model_dir / "preprocessing.joblib"
        metadata_path = self.model_dir / "model_metadata.json"
        try:
            self.preprocessing = joblib.load(preprocessing_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_order = tuple(metadata["feature_order"])
            if metadata_order != FEATURE_ORDER:
                raise ModelArtifactError("Model metadata feature order does not match the API.")
            model_files = metadata["model_files"]
            binary_name = model_files["binary"]
            multiclass_name = model_files["multiclass"]
            if any(
                Path(name).name != name or Path(name).suffix != ".keras"
                for name in (binary_name, multiclass_name)
            ):
                raise ModelArtifactError("Model metadata must reference local .keras artifacts.")
            self.binary_model = tf.keras.models.load_model(
                self.model_dir / binary_name, compile=False, safe_mode=True
            )
            self.multiclass_model = tf.keras.models.load_model(
                self.model_dir / multiclass_name,
                compile=False,
                safe_mode=True,
            )
        except ModelArtifactError:
            raise
        except Exception as exc:
            raise ModelArtifactError("Could not load trained inference artifacts.") from exc

        self.feature_order = tuple(self.preprocessing.get("feature_order", ()))
        if self.feature_order != FEATURE_ORDER:
            raise ModelArtifactError(
                f"Preprocessing feature order mismatch: expected {FEATURE_ORDER}, "
                f"found {self.feature_order}"
            )
        self.scaler = self.preprocessing.get("scaler")
        if self.scaler is None or getattr(self.scaler, "n_features_in_", None) != len(
            FEATURE_ORDER
        ):
            raise ModelArtifactError(
                "Preprocessing artifact does not contain a fitted seven-feature scaler"
            )
        self.model_version = metadata.get("model_version", "unknown")
        if tuple(self.binary_model.input_shape[1:]) != (1, 7) or tuple(
            self.binary_model.output_shape[1:]
        ) != (1,):
            raise ModelArtifactError(
                "Binary model does not match the expected (1, 7) -> (1,) contract."
            )
        if tuple(self.multiclass_model.input_shape[1:]) != (1, 7) or tuple(
            self.multiclass_model.output_shape[1:]
        ) != (7,):
            raise ModelArtifactError(
                "Multiclass model does not match the expected (1, 7) -> (7,) contract."
            )

    def predict(self, features: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(features, Mapping):
            raise InferenceValidationError(
                "invalid_features", "Features must be supplied as a mapping keyed by feature name."
            )
        supplied = set(features)
        expected = set(self.feature_order)
        missing = sorted(expected - supplied)
        extra = sorted(str(name) for name in supplied - expected)
        if missing or extra:
            raise InferenceValidationError(
                "invalid_features",
                "Exactly the seven trained feature names are required.",
                {
                    "missing": missing,
                    "unexpected": extra,
                    "feature_order": list(self.feature_order),
                },
            )

        ordered: list[float] = []
        for name in self.feature_order:
            value = features[name]
            if isinstance(value, bool):
                raise InferenceValidationError(
                    "invalid_features",
                    f"Feature {name} must be a finite number.",
                    {"feature": name},
                )
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise InferenceValidationError(
                    "invalid_features",
                    f"Feature {name} must be a finite number.",
                    {"feature": name},
                ) from exc
            if not math.isfinite(number):
                raise InferenceValidationError(
                    "invalid_features",
                    f"Feature {name} must be a finite number.",
                    {"feature": name},
                )
            ordered.append(number)

        return self.predict_batch(np.asarray([ordered], dtype=np.float64))[0]

    def predict_batch(self, values: object) -> list[dict[str, object]]:
        """Run both serving models once over an ordered (n, 7) feature matrix."""
        raw = np.asarray(values, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != len(self.feature_order) or not len(raw):
            raise InferenceValidationError(
                "invalid_features", "Batch input must have shape (n, 7) with n greater than zero."
            )
        if not np.isfinite(raw).all():
            raise InferenceValidationError("invalid_features", "Features must be finite numbers.")
        scaled = (
            self.scaler.transform(raw).reshape((-1, 1, len(self.feature_order))).astype(np.float32)
        )
        if scaled.shape != (len(raw), 1, 7) or not np.isfinite(scaled).all():
            raise ModelArtifactError("Preprocessing returned invalid model input values.")
        binary_output = np.asarray(self.binary_model.predict(scaled, verbose=0), dtype=np.float64)
        if binary_output.shape not in ((len(raw), 1), (len(raw),)):
            raise ModelArtifactError(
                f"Binary model returned unexpected shape {binary_output.shape}"
            )
        fault_probabilities = binary_output.reshape(-1)
        if not np.isfinite(fault_probabilities).all() or np.any(
            (fault_probabilities < 0) | (fault_probabilities > 1)
        ):
            raise ModelArtifactError("Binary model returned an invalid probability")

        class_output = np.asarray(
            self.multiclass_model.predict(scaled, verbose=0), dtype=np.float64
        )
        if class_output.shape != (len(raw), 7):
            raise ModelArtifactError(
                f"Multiclass model returned unexpected shape {class_output.shape}"
            )
        if not np.isfinite(class_output).all():
            raise ModelArtifactError("Multiclass model returned invalid class probabilities")
        if (
            np.any(class_output < 0.0)
            or np.any(class_output > 1.0)
            or not np.allclose(class_output.sum(axis=1), 1.0, atol=1e-3)
        ):
            raise ModelArtifactError("Multiclass model returned invalid class probabilities")
        results = []
        for fault_probability, probabilities in zip(fault_probabilities, class_output, strict=True):
            fault_probability = float(fault_probability)
            overall_top_code = int(np.argmax(probabilities)) + 1
            results.append(
                {
                    "model_version": self.model_version,
                    "fault_probability": fault_probability,
                    "binary_prediction": "Fault"
                    if fault_probability >= BINARY_FAULT_THRESHOLD
                    else "Normal",
                    "threshold": BINARY_FAULT_THRESHOLD,
                    "class_id": overall_top_code,
                    "predicted_cause": CLASS_MAPPING[overall_top_code]["code"],
                    "confidence": float(probabilities[overall_top_code - 1]),
                    "class_probabilities": {
                        str(code): float(probabilities[code - 1]) for code in CLASS_MAPPING
                    },
                }
            )
        return results
