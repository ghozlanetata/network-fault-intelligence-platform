from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from app.domain import FaultCause
from app.ml.data import (
    CLASS_MAPPING,
    FEATURE_ORDER,
    ValidatedDataset,
    feature_group_ids,
    fit_training_preprocessor,
    load_dataset,
    split_dataset,
    split_summary,
)
from app.ml.inference import InferenceValidationError, ModelInference
from app.ml.models import build_binary_model, build_multiclass_model
from app.ml.train import evaluate_multiclass
from app.recommendations import recommendations_for


def make_dataset(rows_per_class: int = 8, repeats: int = 2) -> ValidatedDataset:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for class_code in range(1, 8):
        for group_index in range(rows_per_class):
            base = class_code * 1000 + group_index * 10
            vector = np.asarray(
                [base + feature_index for feature_index in range(7)], dtype=np.float64
            )
            for _ in range(repeats):
                rows.append(vector.copy())
                labels.append(class_code)
    features = np.asarray(rows, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    groups = feature_group_ids(features)
    frame = pd.DataFrame(features, columns=FEATURE_ORDER)
    frame["FaultCause"] = label_array
    return ValidatedDataset(
        path=Path("synthetic.csv"),
        frame=frame,
        features=features,
        labels=label_array,
        binary_labels=np.where(label_array == 7, 0, 1),
        groups=groups,
    )


def csv_rows() -> pd.DataFrame:
    rows = []
    for class_code in range(1, 8):
        base = class_code * 10.0
        rows.append(
            {
                "cell": f"cell-{class_code}",
                "lon": 2.0,
                "lat": 36.0,
                "[Retainability": base + 0.1,
                "HOSR": base + 0.2,
                "RSRP": base + 0.3,
                "RSRQ": base + 0.4,
                "SINR": base + 0.5,
                "Throughput": base + 0.6,
                "Distance": base + 0.7,
                "FaultCause]": class_code,
            }
        )
    return pd.DataFrame(rows)


def test_dataset_header_normalization_numeric_validation_and_binary_target(tmp_path: Path) -> None:
    path = tmp_path / "db_fm_validation14.csv"
    csv_rows().to_csv(path, index=False)
    dataset = load_dataset(path)

    assert tuple(name for name in FEATURE_ORDER if name in dataset.frame.columns) == FEATURE_ORDER
    assert dataset.features.shape == (7, 7)
    assert np.isfinite(dataset.features).all()
    assert set(np.unique(dataset.labels)) == set(range(1, 8))
    np.testing.assert_array_equal(dataset.binary_labels, [1, 1, 1, 1, 1, 1, 0])
    assert dataset.path == path.resolve()


def test_dataset_rejects_missing_values_and_invalid_labels(tmp_path: Path) -> None:
    missing_path = tmp_path / "db_fm_validation14.csv"
    frame = csv_rows()
    frame.loc[0, "RSRP"] = np.nan
    frame.to_csv(missing_path, index=False)
    with pytest.raises(ValueError, match="missing or non-finite"):
        load_dataset(missing_path)

    frame = csv_rows()
    frame.loc[0, "FaultCause]"] = 8
    frame.to_csv(missing_path, index=False)
    with pytest.raises(ValueError, match="contain every class"):
        load_dataset(missing_path)


def test_group_ids_are_deterministic_and_identical_vectors_match() -> None:
    features = np.asarray([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]], dtype=float)
    first = feature_group_ids(features)
    second = feature_group_ids(features.copy())
    assert first[0] == first[1]
    np.testing.assert_array_equal(first, second)


def test_split_is_deterministic_stratified_and_has_no_group_leakage() -> None:
    dataset = make_dataset()
    split = split_dataset(dataset, seed=42)
    repeat = split_dataset(dataset, seed=42)
    np.testing.assert_array_equal(split.train, repeat.train)
    np.testing.assert_array_equal(split.validation, repeat.validation)
    np.testing.assert_array_equal(split.test, repeat.test)
    summary = split_summary(dataset, split)
    assert summary["group_leakage"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
        "all_groups_assigned_once": True,
    }
    for split_name in ("train", "validation", "test"):
        assert set(summary[split_name]["class_distribution"].values())
        assert all(int(value) > 0 for value in summary[split_name]["class_distribution"].values())


def test_scaler_is_fit_on_train_partition_only() -> None:
    train = np.asarray([[0, 1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7, 8]], dtype=float)
    validation = np.asarray([[100, 100, 100, 100, 100, 100, 100]], dtype=float)
    test = np.asarray([[-100, -100, -100, -100, -100, -100, -100]], dtype=float)
    scaler, x_train, x_validation, x_test = fit_training_preprocessor(train, validation, test)
    np.testing.assert_allclose(scaler.mean_, train.mean(axis=0))
    assert scaler.n_samples_seen_ == len(train)
    assert x_train.shape == (2, 1, 7)
    assert x_validation.shape == x_test.shape == (1, 1, 7)
    assert np.all(np.abs(x_validation) > 1)


def test_new_models_save_load_and_emit_expected_probabilities(tmp_path: Path) -> None:
    inputs = np.zeros((3, 1, 7), dtype=np.float32)
    binary_path = tmp_path / "binary.keras"
    multiclass_path = tmp_path / "multiclass.keras"
    binary = build_binary_model()
    multiclass = build_multiclass_model()
    assert binary.output_shape == (None, 1)
    assert multiclass.output_shape == (None, 7)
    binary.save(binary_path)
    multiclass.save(multiclass_path)
    binary_loaded = tf.keras.models.load_model(binary_path, compile=False, safe_mode=True)
    multiclass_loaded = tf.keras.models.load_model(multiclass_path, compile=False, safe_mode=True)
    binary_probs = binary_loaded.predict(inputs, verbose=0)
    multiclass_probs = multiclass_loaded.predict(inputs, verbose=0)
    assert binary_probs.shape == (3, 1)
    assert np.isfinite(binary_probs).all()
    assert np.all((binary_probs >= 0) & (binary_probs <= 1))
    assert multiclass_probs.shape == (3, 7)
    assert np.isfinite(multiclass_probs).all()
    np.testing.assert_allclose(multiclass_probs.sum(axis=1), 1.0, atol=1e-5)


def test_multiclass_evaluation_preserves_recovered_one_based_labels() -> None:
    labels = np.arange(1, 8, dtype=np.int64)
    probabilities = np.eye(7, dtype=np.float64)

    class FixedModel:
        def predict(self, values: np.ndarray, verbose: int = 0) -> np.ndarray:
            del verbose
            assert len(values) == len(labels)
            return probabilities

    metrics = evaluate_multiclass(FixedModel(), np.zeros((7, 1, 7)), labels)
    assert metrics["accuracy"] == 1.0
    assert [entry["support"] for entry in metrics["per_class"].values()] == [1] * 7


class FakePredictor:
    def __init__(self, output: np.ndarray):
        self.output = output
        self.calls = 0
        self.input = None
        self.input_shape = (None, 1, 7)
        self.output_shape = (None, *output.shape[1:])

    def predict(self, values: np.ndarray, verbose: int = 0) -> np.ndarray:
        del verbose
        self.calls += 1
        self.input = values.copy()
        return self.output.copy()


def make_inference(tmp_path: Path, monkeypatch, binary_output: list[list[float]]):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    training_values = np.arange(70, dtype=float).reshape(10, 7)
    scaler = StandardScaler().fit(training_values)
    joblib.dump(
        {"feature_order": list(FEATURE_ORDER), "scaler": scaler}, model_dir / "preprocessing.joblib"
    )
    (model_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "model_version": "test-model",
                "feature_order": list(FEATURE_ORDER),
                "model_files": {"binary": "binary.keras", "multiclass": "multi.keras"},
            }
        ),
        encoding="utf-8",
    )
    binary = FakePredictor(np.asarray(binary_output, dtype=float))
    multi = FakePredictor(np.asarray([[0.1, 0.7, 0.05, 0.05, 0.04, 0.03, 0.03]], dtype=float))

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Inference must never fit the preprocessing scaler")

    monkeypatch.setattr(StandardScaler, "fit", forbidden_fit)
    import app.ml.inference as inference_module

    def fake_load_model(path, **kwargs):
        del kwargs
        return binary if Path(path).name == "binary.keras" else multi

    monkeypatch.setattr(inference_module.tf.keras.models, "load_model", fake_load_model)
    service = ModelInference(model_dir)
    return service, scaler, binary, multi


def test_inference_normal_still_classifies_and_never_refits(tmp_path: Path, monkeypatch) -> None:
    service, scaler, binary, multiclass = make_inference(tmp_path, monkeypatch, [[0.2]])
    # Construct deliberately in reverse insertion order; the persisted feature order wins.
    sample = {name: float(FEATURE_ORDER.index(name) + 2) for name in reversed(FEATURE_ORDER)}
    result = service.predict(sample)
    ordered = np.asarray([[float(index + 2) for index in range(7)]])
    expected = scaler.transform(ordered).reshape((1, 1, 7)).astype(np.float32)
    np.testing.assert_allclose(binary.input, expected)
    assert result["binary_prediction"] == "Normal"
    assert result["class_id"] == 2
    assert result["predicted_cause"] == "CH"
    assert len(result["class_probabilities"]) == 7
    assert multiclass.calls == 1


def test_inference_fault_returns_valid_cause_and_probabilities(tmp_path: Path, monkeypatch) -> None:
    service, _, binary, multiclass = make_inference(tmp_path, monkeypatch, [[0.8]])
    sample = {name: float(index) for index, name in enumerate(FEATURE_ORDER)}
    result = service.predict(sample)
    assert binary.calls == multiclass.calls == 1
    assert result["binary_prediction"] == "Fault"
    assert result["class_id"] == 2
    assert result["predicted_cause"] == "CH"
    assert set(result["class_probabilities"]) == {str(code) for code in CLASS_MAPPING}


def test_batch_inference_transforms_seven_features_and_invokes_models_once(
    tmp_path: Path, monkeypatch
) -> None:
    service, scaler, binary, multiclass = make_inference(tmp_path, monkeypatch, [[0.2]])
    values = np.arange(21, dtype=float).reshape(3, 7)
    binary.output = np.asarray([[0.2], [0.8], [0.1]])
    multiclass.output = np.tile(np.asarray([[0.1, 0.7, 0.05, 0.05, 0.04, 0.03, 0.03]]), (3, 1))
    results = service.predict_batch(values)
    expected = scaler.transform(values).reshape((3, 1, 7)).astype(np.float32)
    np.testing.assert_allclose(binary.input, expected)
    np.testing.assert_allclose(multiclass.input, expected)
    assert len(results) == 3
    assert binary.calls == multiclass.calls == 1


def test_inference_preserves_binary_multiclass_disagreement(tmp_path: Path, monkeypatch) -> None:
    service, _, _, multiclass = make_inference(tmp_path, monkeypatch, [[0.8]])
    multiclass.output = np.asarray([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.94]])
    sample = {name: float(index) for index, name in enumerate(FEATURE_ORDER)}
    result = service.predict(sample)
    assert result["binary_prediction"] == "Fault"
    assert result["class_id"] == 7
    assert result["predicted_cause"] == "Normal"
    assert result["class_probabilities"]["7"] == pytest.approx(0.94)


def test_inference_invalid_input_has_structured_error(tmp_path: Path, monkeypatch) -> None:
    service, *_ = make_inference(tmp_path, monkeypatch, [[0.2]])
    sample = {name: float(index) for index, name in enumerate(FEATURE_ORDER)}
    sample["RSRP"] = float("nan")
    with pytest.raises(InferenceValidationError) as captured:
        service.predict(sample)
    assert captured.value.as_dict()["code"] == "invalid_features"
    sample.pop("RSRP")
    with pytest.raises(InferenceValidationError) as captured:
        service.predict(sample)
    assert "missing" in captured.value.as_dict()["details"]


def test_recommendations_are_exact_for_every_cause() -> None:
    expected = {
        FaultCause.ED: [
            "Check Bandwidth Capacity Configuration",
            "Check Frequency Configuration",
            "Check Tilt Configuration",
            "Check Load Balance",
        ],
        FaultCause.CH: [
            "Check Power Configuration",
            "Check Cell Range",
            "Check Tilt Configuration",
        ],
        FaultCause.II: [
            "Check Planification of Sites",
            "Check Frequency Configuration",
            "Check Coverage",
        ],
        FaultCause.TLHO: ["Check Handover Execution"],
        FaultCause.RP: ["Check if RRU is Faulty", "Check Licence", "Check Power on Site"],
        FaultCause.EU: [
            "Check Bandwidth Capacity Configuration",
            "Check Frequency Configuration",
            "Check Tilt Configuration",
            "Check Load Balance",
            "Check Synchronization of Cell",
        ],
        FaultCause.NORMAL: [],
    }
    assert set(expected) == set(FaultCause)
    for cause, actions in expected.items():
        assert recommendations_for(cause) == actions
