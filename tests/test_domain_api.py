import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import get_settings
from app.domain import FaultCause, PredictionStatus
from app.inference import (
    ArtifactInference,
    InferenceResult,
    resolve_status,
)
from app.ml.data import FEATURE_ORDER, ROOT, load_dataset, locate_dataset
from app.recommendations import recommendations_for
from app.schemas import KPIRecord


def result_for(
    status_value: PredictionStatus = PredictionStatus.FAULT,
    *,
    binary: str = "Fault",
    class_id: int = 2,
    cause: str | None = "CH",
) -> InferenceResult:
    predicted_cause = {1: "ED", 2: "CH", 3: "II", 4: "TLHO", 5: "RP", 6: "EU", 7: "Normal"}[
        class_id
    ]
    return InferenceResult(
        status=status_value,
        fault=True
        if status_value == PredictionStatus.FAULT
        else False
        if status_value == PredictionStatus.NORMAL
        else None,
        cause=cause,
        cause_name="Coverage Hole" if cause == "CH" else "Normal" if cause == "Normal" else None,
        binary_prediction=binary,
        fault_probability=0.8 if binary == "Fault" else 0.2,
        normal_probability=0.2 if binary == "Fault" else 0.8,
        threshold=0.5,
        class_id=class_id,
        predicted_cause=predicted_cause,
        confidence=0.7 if class_id != 7 else 0.94,
        class_probabilities={
            "ED": 0.01,
            "CH": 0.7 if class_id == 2 else 0.01,
            "II": 0.01,
            "TLHO": 0.01,
            "RP": 0.01,
            "EU": 0.01,
            "Normal": 0.94 if class_id == 7 else 0.25,
        },
        model_version="test-model-v1",
    )


class StubInference:
    available = True

    def __init__(self, result: InferenceResult | None = None):
        self.result = result or result_for()
        self.request = None

    def predict(self, request: KPIRecord) -> InferenceResult:
        self.request = request
        return self.result

    def predict_batch(self, requests: list[KPIRecord]) -> list[InferenceResult]:
        self.batch = requests
        return [self.result for _ in requests]


def external_network_health_dataset_missing() -> bool:
    try:
        locate_dataset()
    except FileNotFoundError:
        return True
    return False


NETWORK_HEALTH_DATASET_MISSING = external_network_health_dataset_missing()


def open_client(
    monkeypatch, tmp_path: Path, inference, bootstrap_role: str = "operator"
) -> TestClient:
    db_path = tmp_path / "history.sqlite3"
    monkeypatch.setenv("NFI_DATABASE_PATH", str(db_path))
    monkeypatch.setenv("NFI_BOOTSTRAP_USERNAME", "operator")
    monkeypatch.setenv("NFI_BOOTSTRAP_PASSWORD", "test-password")
    monkeypatch.setenv("NFI_BOOTSTRAP_ROLE", bootstrap_role)
    monkeypatch.setattr(main_module, "ArtifactInference", lambda _model_dir: inference)
    get_settings.cache_clear()
    return AuthenticatedClient(main_module.app)


class AuthenticatedClient:
    def __init__(self, app):
        self.client = TestClient(app)

    def __enter__(self):
        self.client.__enter__()
        response = self.client.post(
            "/v1/auth/login", json={"username": "operator", "password": "test-password"}
        )
        assert response.status_code == 200
        return self.client

    def __exit__(self, *args):
        return self.client.__exit__(*args)


def valid_payload() -> dict[str, float]:
    return {
        "retainability": 0.99,
        "hosr": 0.935,
        "rsrp": -71.332,
        "rsrq": -18.608,
        "sinr": 14.386,
        "average_throughput": 98.834,
        "distance": 0.87,
    }


def assign_role(database: Path, role: str) -> None:
    import sqlite3

    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role = ?", (role,))


def test_original_recommendation_rules_are_preserved() -> None:
    assert recommendations_for(FaultCause.ED) == [
        "Check Bandwidth Capacity Configuration",
        "Check Frequency Configuration",
        "Check Tilt Configuration",
        "Check Load Balance",
    ]
    assert recommendations_for(FaultCause.NORMAL) == []
    assert recommendations_for(FaultCause.EU)[-1] == "Check Synchronization of Cell"


def test_recommendation_endpoint_and_readiness(monkeypatch, tmp_path: Path) -> None:
    with open_client(monkeypatch, tmp_path, StubInference()) as client:
        response = client.post("/v1/recommendations", json={"cause": "RP"})
        readiness = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["engine"] == "deterministic_rules"
    assert response.json()["actions"] == [
        "Check if RRU is Faulty",
        "Check Licence",
        "Check Power on Site",
    ]
    assert readiness.status_code == 200
    assert readiness.json() == {
        "status": "ready",
        "database": "ready",
        "inference": "ready",
    }


def test_operator_ui_and_assets_are_served_by_fastapi(monkeypatch, tmp_path: Path) -> None:
    with open_client(monkeypatch, tmp_path, StubInference()) as client:
        page = client.get("/")
        stylesheet = client.get("/assets/app.css")
        script = client.get("/assets/app.js")
    assert page.status_code == stylesheet.status_code == script.status_code == 200
    assert "Network Fault Intelligence" in page.text
    assert "INFERENCE ONLINE" in script.text
    assert ".sidebar" in stylesheet.text


def test_valid_seven_feature_request_returns_fault_and_persists_history(
    monkeypatch, tmp_path: Path
) -> None:
    inference = StubInference(result_for())
    with open_client(monkeypatch, tmp_path, inference) as client:
        response = client.post("/v1/predictions", json=valid_payload())
        history = client.get("/v1/history")
    assert response.status_code == 201
    body = response.json()
    assert body["schema_version"] == "v1"
    assert body["status"] == "FAULT"
    assert body["fault"] is True
    assert body["cause"] == "CH"
    assert body["kpis"] == valid_payload()
    assert not {"binary_detection", "classification", "model_version"} & body.keys()
    assert body["recommendations"] == recommendations_for(FaultCause.CH)
    spoofed = client.post(
        "/v1/predictions",
        json={**valid_payload(), "role": "platform_ml_admin"},
        headers={"X-Role": "platform_ml_admin"},
    )
    assert spoofed.status_code == 422
    header_spoof = client.post(
        "/v1/predictions",
        json=valid_payload(),
        headers={"X-Role": "platform_ml_admin"},
    ).json()
    assert "classification" not in header_spoof
    assert "model_version" not in header_spoof
    assert inference.request.cell is None
    assert history.status_code == 200
    saved = history.json()[0]
    assert saved["status"] == "FAULT"
    assert {key: saved["kpis"][key] for key in valid_payload()} == valid_payload()
    assert not {"model_version"} & saved.keys()
    assert not {"binary_detection", "classification"} & saved["inference"].keys()
    assign_role(client.app.state.auth.path, "network_admin")
    network_admin_result = client.post("/v1/predictions", json=valid_payload()).json()
    assert "classification" not in network_admin_result
    network_admin_history = client.get("/v1/history").json()[0]
    assert "model_version" not in network_admin_history
    assign_role(client.app.state.auth.path, "platform_ml_admin")
    privileged_prediction = client.post("/v1/predictions", json=valid_payload()).json()
    privileged_history = client.get("/v1/history").json()[0]
    assert privileged_prediction["classification"]["probabilities"]["CH"] == 0.7
    assert privileged_prediction["binary_detection"]["fault_probability"] == 0.8
    assert privileged_history["model_version"] == "test-model-v1"
    assert privileged_history["inference"]["classification"]["probabilities"]["CH"] == 0.7


@pytest.mark.skipif(
    NETWORK_HEALTH_DATASET_MISSING,
    reason="Network Health external dataset is not available.",
)
def test_network_health_is_filtered_for_operator_and_full_for_platform_ml_admin(
    monkeypatch, tmp_path: Path
) -> None:
    inference = StubInference(result_for())
    with open_client(monkeypatch, tmp_path, inference) as client:
        response = client.post("/v1/network-health/runs", json={})
        latest = client.get("/v1/network-health/latest")
        results = client.get("/v1/network-health/results?limit=1200")
        history = client.get("/v1/history")
        manual = client.post("/v1/predictions", json=valid_payload())
        history_after = client.get("/v1/history")
    assert response.status_code == 201
    assert latest.json()["status"] == "COMPLETED"
    assert latest.json()["record_count"] == 1137
    assert latest.json()["summary"]["total_cells_analyzed"] == 1137
    assert len(inference.batch) == 1137
    assert results.json()["total"] == 1137
    sample = results.json()["items"][0]
    assert sample["predicted_cause"] == "CH"
    assert len(sample["kpis"]) == 7
    assert sample["recommendations"] == recommendations_for(FaultCause.CH)
    restricted = {
        "ground_truth_cause", "ground_truth_cause_code", "ground_truth_cause_name",
        "binary_probability", "binary_normal_probability", "multiclass_probabilities",
        "classifier_predicted_cause", "model_version",
    }
    assert not restricted & sample.keys()
    assert "validation_dataset_performance" not in latest.json()["summary"]
    assert "model_version" not in latest.json()
    assert "dataset_name" not in latest.json()
    assert history.json() == []
    assert manual.status_code == 201
    assert len(history_after.json()) == 1
    assign_role(client.app.state.auth.path, "platform_ml_admin")
    privileged = client.get("/v1/network-health/results?limit=1").json()["items"][0]
    assert privileged["ground_truth_cause"] in range(1, 8)
    assert privileged["classifier_predicted_cause"] == "CH"
    assert privileged["multiclass_probabilities"]["CH"] == 0.7
    assert privileged["model_version"] == "test-model-v1"
    admin_summary = client.get("/v1/network-health/latest").json()["summary"]
    assert "validation_dataset_performance" in admin_summary


def test_network_health_rejects_a_second_active_dataset_run(monkeypatch, tmp_path: Path) -> None:
    with open_client(monkeypatch, tmp_path, StubInference()) as client:
        active_run_id = client.app.state.network_health.create_run("db_fm_validation14.csv")
        response = client.post("/v1/network-health/runs", json={})
        latest = client.get("/v1/network-health/latest")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "network_health_in_progress"
    assert latest.json()["run_id"] == active_run_id
    assert latest.json()["status"] == "PROCESSING"


@pytest.mark.skipif(
    NETWORK_HEALTH_DATASET_MISSING,
    reason="Network Health external dataset is not available.",
)
def test_selected_network_health_cell_returns_same_operator_investigation_record(
    monkeypatch, tmp_path: Path
) -> None:
    with open_client(monkeypatch, tmp_path, StubInference()) as client:
        run = client.post("/v1/network-health/runs", json={})
        all_results = client.get("/v1/network-health/results?limit=5000").json()
        assert run.status_code == 201
        assert all_results["total"] == 1137
        selected = next(row for row in all_results["items"] if row["status"] == "FAULT")

        # Network Sites / Site Investigation fetches the selected cell from this same run.
        from urllib.parse import quote

        investigation = client.get(
            f"/v1/network-health/results?limit=5000&cell={quote(selected['cell'])}"
        ).json()
        matching = next(row for row in investigation["items"] if row["cell"] == selected["cell"])
        assert matching["cell"] == selected["cell"]
        assert matching["kpis"] == selected["kpis"]
        assert matching["predicted_cause"] == selected["predicted_cause"]
        expected_actions = recommendations_for(FaultCause(selected["predicted_cause"]))
        assert matching["recommendations"] == expected_actions
        assert len(matching["kpis"]) == 7
        assert not {
            "ground_truth_cause", "ground_truth_cause_code", "ground_truth_cause_name",
            "binary_probability", "binary_normal_probability", "multiclass_probabilities",
            "classifier_predicted_cause", "model_version",
        } & matching.keys()

        assign_role(client.app.state.auth.path, "network_admin")
        network_admin = client.get(
            f"/v1/network-health/results?limit=5000&cell={quote(selected['cell'])}"
        ).json()
        admin_matching = next(
            row for row in network_admin["items"] if row["cell"] == selected["cell"]
        )
        assert admin_matching["cell"] == selected["cell"]
        assert admin_matching["kpis"] == selected["kpis"]
        assert admin_matching["predicted_cause"] == selected["predicted_cause"]
        assert admin_matching["recommendations"] == matching["recommendations"]
        assert not {
            "ground_truth_cause", "binary_probability", "multiclass_probabilities",
            "classifier_predicted_cause", "model_version",
        } & admin_matching.keys()


@pytest.mark.skipif(
    NETWORK_HEALTH_DATASET_MISSING,
    reason="Network Health external dataset is not available.",
)
def test_network_health_real_serving_artifacts_end_to_end(
    monkeypatch, tmp_path: Path
) -> None:
    dataset = load_dataset()
    assert len(dataset.frame) == 1137
    db_path = tmp_path / "real-network-health.sqlite3"
    monkeypatch.setenv("NFI_DATABASE_PATH", str(db_path))
    monkeypatch.setenv("NFI_BOOTSTRAP_USERNAME", "operator")
    monkeypatch.setenv("NFI_BOOTSTRAP_PASSWORD", "test-password")
    monkeypatch.setenv("NFI_BOOTSTRAP_ROLE", "platform_ml_admin")
    get_settings.cache_clear()
    with TestClient(main_module.app) as client:
        login = client.post(
            "/v1/auth/login", json={"username": "operator", "password": "test-password"}
        )
        assert login.status_code == 200
        run = client.post("/v1/network-health/runs", json={})
        assert run.status_code == 201, run.text
        latest = client.get("/v1/network-health/latest").json()
        results = client.get("/v1/network-health/results?limit=1200").json()
        history = client.get("/v1/history").json()
    summary = latest["summary"]
    assert latest["status"] == "COMPLETED"
    assert latest["model_version"] == "nfi-retrained-lstm-v1"
    assert latest["record_count"] == summary["total_cells_analyzed"] == 1137
    assert results["total"] == len(results["items"]) == 1137
    assert (
        summary["normal_count"] + summary["fault_count"] + summary["review_required_count"] == 1137
    )
    assert sum(summary["predicted_cause_distribution"].values()) == 1137
    assert summary["validation_dataset_performance"]["dataset_records"] == 1137
    assert summary["normal_count"] == sum(row["status"] == "NORMAL" for row in results["items"])
    assert summary["fault_count"] == sum(row["status"] == "FAULT" for row in results["items"])
    assert summary["review_required_count"] == sum(
        row["status"] == "REVIEW_REQUIRED" for row in results["items"]
    )
    for index in (0, len(dataset.frame) // 2, len(dataset.frame) - 1):
        source = dataset.frame.iloc[index]
        result = results["items"][index]
        assert result["cell"] == str(source["cell"])
        assert result["latitude"] == pytest.approx(float(source["lat"]))
        assert result["longitude"] == pytest.approx(float(source["lon"]))
        assert tuple(result["kpis"]) == FEATURE_ORDER
        assert tuple(result["kpis"].values()) == pytest.approx(dataset.features[index])
        assert result["ground_truth_cause"] == int(dataset.labels[index])
        assert result["status"] in {status.value for status in PredictionStatus}
        binary_fault = result["binary_prediction"] == "Fault"
        classifier_fault = result["classifier_predicted_cause"] != "Normal"
        expected_status = (
            PredictionStatus.FAULT
            if binary_fault and classifier_fault
            else PredictionStatus.NORMAL
            if not binary_fault and not classifier_fault
            else PredictionStatus.REVIEW_REQUIRED
        )
        assert result["status"] == expected_status.value
        assert result["predicted_cause"] is None or isinstance(result["predicted_cause"], str)
        assert isinstance(result["recommendations"], list)
        assert len(result["multiclass_probabilities"]) == 7
        assert result["model_version"] == "nfi-retrained-lstm-v1"
    assert history == []
    import sqlite3

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body: body.pop("sinr"),
        lambda body: body.update(hosr="not-a-number"),
        lambda body: body.update(rsrp=float("nan")),
        lambda body: body.update(distance=float("inf")),
        lambda body: body.update(retainability=True),
        lambda body: body.update(distance=-0.1),
        lambda body: body.update(unexpected=1),
    ],
)
def test_invalid_requests_return_structured_422(mutate, monkeypatch, tmp_path: Path) -> None:
    body = valid_payload()
    mutate(body)
    with open_client(monkeypatch, tmp_path, StubInference()) as client:
        response = client.post(
            "/v1/predictions",
            content=json.dumps(body, allow_nan=True),
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_malformed_json_returns_422(monkeypatch, tmp_path: Path) -> None:
    with open_client(monkeypatch, tmp_path, StubInference()) as client:
        response = client.post(
            "/v1/predictions", content="{", headers={"content-type": "application/json"}
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    ("binary", "class_id", "expected"),
    [
        ("Normal", 7, PredictionStatus.NORMAL),
        ("Fault", 4, PredictionStatus.FAULT),
        ("Fault", 7, PredictionStatus.REVIEW_REQUIRED),
        ("Normal", 2, PredictionStatus.REVIEW_REQUIRED),
    ],
)
def test_consistency_policy(binary, class_id, expected) -> None:
    assert resolve_status(binary, class_id) == expected


@pytest.mark.parametrize(
    ("result", "expected_status"),
    [
        (
            result_for(PredictionStatus.NORMAL, binary="Normal", class_id=7, cause="Normal"),
            "NORMAL",
        ),
        (result_for(PredictionStatus.REVIEW_REQUIRED, class_id=7, cause=None), "REVIEW_REQUIRED"),
        (
            result_for(PredictionStatus.REVIEW_REQUIRED, binary="Normal", class_id=2, cause=None),
            "REVIEW_REQUIRED",
        ),
    ],
)
def test_normal_and_review_response_contract(result, expected_status, monkeypatch, tmp_path):
    with open_client(monkeypatch, tmp_path, StubInference(result)) as client:
        response = client.post("/v1/predictions", json=valid_payload())
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == expected_status
    assert body["recommendations"] == []
    if expected_status == "NORMAL":
        assert body["cause"] == "Normal"
        assert body["fault"] is False
    else:
        assert body["cause"] is None
        assert body["fault"] is None
        assert "classification" not in body


def test_missing_model_artifacts_degrade_readiness_and_return_503(
    monkeypatch, tmp_path: Path
) -> None:
    unavailable = ArtifactInference(tmp_path / "missing-models")
    assert unavailable.available is False
    with open_client(monkeypatch, tmp_path, unavailable) as client:
        ready = client.get("/health/ready")
        response = client.post("/v1/predictions", json=valid_payload())
        history = client.get("/v1/history")
    assert ready.status_code == 200
    assert ready.json()["inference"] == "artifacts_unavailable"
    assert ready.json()["status"] == "degraded"
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_artifacts_unavailable"
    assert history.json() == []


def test_openapi_documents_inputs_response_and_review_status(monkeypatch, tmp_path: Path) -> None:
    with open_client(
        monkeypatch, tmp_path, StubInference(), bootstrap_role="platform_ml_admin"
    ) as client:
        schema = client.get("/openapi.json").json()
    route = schema["paths"]["/v1/predictions"]["post"]
    assert "REVIEW_REQUIRED" in route["description"]
    request_schema = schema["components"]["schemas"]["PredictionRequest"]
    assert set(request_schema["required"]) == {
        "retainability",
        "hosr",
        "rsrp",
        "rsrq",
        "sinr",
        "average_throughput",
        "distance",
    }
    assert "Platform/ML Admin" in route["description"]
    response_schema = route["responses"]["201"]["content"]["application/json"]["schema"]
    assert {item["$ref"].rsplit("/", 1)[-1] for item in response_schema["anyOf"]} == {
        "OperationalPredictionResponse",
        "PredictionResponse",
    }


def test_docs_require_platform_ml_admin_and_distinguish_401_from_403(monkeypatch, tmp_path: Path):
    with open_client(monkeypatch, tmp_path, StubInference()) as client:
        assert client.get("/docs").status_code == 403
        assign_role(client.app.state.auth.path, "network_admin")
        assert client.get("/openapi.json").status_code == 403
        assign_role(client.app.state.auth.path, "platform_ml_admin")
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200


def test_dataset_error_details_are_hidden_from_operator_and_available_to_ml_admin(
    monkeypatch, tmp_path: Path
) -> None:
    with open_client(monkeypatch, tmp_path, StubInference()) as client:
        monkeypatch.setattr(
            main_module,
            "process_dataset",
            lambda *_args: (_ for _ in ()).throw(ValueError("private /models/train.csv detail")),
        )
        operator_response = client.post("/v1/network-health/runs", json={})
        assign_role(client.app.state.auth.path, "platform_ml_admin")
        admin_response = client.post("/v1/network-health/runs", json={})
    assert operator_response.status_code == admin_response.status_code == 422
    assert "private" not in operator_response.json()["error"]["message"]
    assert "private" in admin_response.json()["error"]["message"]


def test_liveness_does_not_depend_on_model_readiness(monkeypatch, tmp_path: Path) -> None:
    unavailable = ArtifactInference(tmp_path / "absent")
    with open_client(monkeypatch, tmp_path, unavailable) as client:
        assert client.get("/health/live").json() == {"status": "alive"}


def test_database_failure_makes_readiness_not_ready(monkeypatch, tmp_path: Path) -> None:
    with open_client(monkeypatch, tmp_path, StubInference()) as client:
        client.app.state.repository.healthy = lambda: False
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["error"]["readiness"] == {
        "status": "not_ready",
        "database": "unavailable",
        "inference": "ready",
    }


@pytest.mark.skipif(
    not (ROOT / "models" / "binary_fault_detector.keras").is_file()
    or not (ROOT / "models" / "fault_cause_classifier.keras").is_file()
    or not (ROOT / "models" / "preprocessing.joblib").is_file(),
    reason="Ignored local ML artifacts are not installed in this environment.",
)
def test_saved_local_models_load_and_real_inference_never_fits(monkeypatch) -> None:
    from sklearn.preprocessing import StandardScaler

    service = ArtifactInference(ROOT / "models")
    assert service.available
    assert service._engine.binary_model.output_shape == (None, 1)
    assert service._engine.multiclass_model.output_shape == (None, 7)

    def forbidden_fit(*_args, **_kwargs):
        raise AssertionError("The persisted inference scaler must never be fit at request time")

    monkeypatch.setattr(StandardScaler, "fit", forbidden_fit)
    sample = dict(
        zip(FEATURE_ORDER, [0.99, 0.935, -71.332, -18.608, 14.386, 98.834, 0.87], strict=True)
    )
    result = service.predict(
        KPIRecord.model_validate(
            {
                "retainability": sample["Retainability"],
                "hosr": sample["HOSR"],
                "rsrp": sample["RSRP"],
                "rsrq": sample["RSRQ"],
                "sinr": sample["SINR"],
                "average_throughput": sample["Throughput"],
                "distance": sample["Distance"],
            }
        )
    )
    assert isinstance(result.status, PredictionStatus)
    assert 0 <= result.fault_probability <= 1
    assert 0 <= result.normal_probability <= 1
    assert set(result.class_probabilities) == {"ED", "CH", "II", "TLHO", "RP", "EU", "Normal"}
    assert np.isclose(sum(result.class_probabilities.values()), 1.0, atol=1e-5)
