from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main_module
from app.config import get_settings
from app.domain import FaultCause, PredictionStatus
from app.inference import InferenceResult
from app.ml.data import ROOT
from app.schemas import KPIRecord


class StubInference:
    available = True

    def predict(self, _request: KPIRecord) -> InferenceResult:
        return InferenceResult(
            status=PredictionStatus.FAULT,
            fault=True,
            cause="CH",
            cause_name="Coverage Hole",
            binary_prediction="Fault",
            fault_probability=0.8,
            normal_probability=0.2,
            threshold=0.5,
            class_id=2,
            predicted_cause="CH",
            confidence=0.8,
            class_probabilities={
                "ED": 0.02,
                "CH": 0.8,
                "II": 0.03,
                "TLHO": 0.03,
                "RP": 0.03,
                "EU": 0.03,
                "Normal": 0.06,
            },
            model_version="test-model",
        )

    def predict_batch(self, _requests):
        return []


def setup_client(monkeypatch, tmp_path: Path) -> tuple[TestClient, Path]:
    database = tmp_path / "investigations.sqlite3"
    monkeypatch.setenv("NFI_DATABASE_PATH", str(database))
    monkeypatch.setenv("NFI_BOOTSTRAP_USERNAME", "operator")
    monkeypatch.setenv("NFI_BOOTSTRAP_PASSWORD", "test-password")
    monkeypatch.setenv("NFI_BOOTSTRAP_ROLE", "operator")
    monkeypatch.setenv("NFI_MODEL_DIR", str(tmp_path / "missing-models"))
    monkeypatch.setattr(main_module, "ArtifactInference", lambda _path: StubInference())
    get_settings.cache_clear()
    client = TestClient(main_module.app)
    client.__enter__()
    response = client.post(
        "/v1/auth/login", json={"username": "operator", "password": "test-password"}
    )
    assert response.status_code == 200
    return client, database


def prediction(client: TestClient) -> int:
    response = client.post(
        "/v1/predictions",
        json={
            "retainability": 0.99,
            "hosr": 0.935,
            "rsrp": -71.3,
            "rsrq": -18.6,
            "sinr": 14.3,
            "average_throughput": 98.8,
            "distance": 0.87,
            "cell": "TEST-CELL",
            "latitude": 36.6,
            "longitude": 3.1,
        },
    )
    assert response.status_code == 201
    return response.json()["prediction_id"]


def create_investigation(client: TestClient) -> tuple[int, int]:
    prediction_id = prediction(client)
    response = client.post("/v1/investigations", json={"prediction_id": prediction_id})
    assert response.status_code == 201, response.text
    return prediction_id, response.json()["investigation_id"]


def submit(client: TestClient, investigation_id: int, cause: str | None = "CH") -> None:
    updated = client.patch(
        f"/v1/investigations/{investigation_id}",
        json={
            "findings": "Field inspection recorded evidence for review.",
            "investigation_cause": cause,
            "resolution_notes": "Corrective work recorded.",
            "is_completed": True,
        },
    )
    assert updated.status_code == 200, updated.text
    response = client.post(f"/v1/investigations/{investigation_id}/submit", json={})
    assert response.status_code == 200, response.text


def test_unauthenticated_investigation_access_is_denied(monkeypatch, tmp_path):
    client, _ = setup_client(monkeypatch, tmp_path)
    client.cookies.clear()
    assert client.get("/v1/investigations").status_code == 401
    assert client.get("/v1/investigations/1").status_code == 401
    assert client.post("/v1/investigations", json={"prediction_id": 1}).status_code == 401
    client.__exit__(None, None, None)


def test_operator_create_update_submit_and_cannot_manufacture_verification(monkeypatch, tmp_path):
    client, _ = setup_client(monkeypatch, tmp_path)
    _, investigation_id = create_investigation(client)
    assert client.get(f"/v1/investigations/{investigation_id}").status_code == 200
    patch = client.patch(
        f"/v1/investigations/{investigation_id}",
        json={
            "findings": "Observed evidence supports CH.",
            "investigation_cause": "CH",
            "is_completed": True,
        },
    )
    assert patch.status_code == 200
    assert patch.json()["investigation_status"] == "IN_PROGRESS"
    assert client.post(f"/v1/investigations/{investigation_id}/submit", json={}).status_code == 200
    assert (
        client.post(
            f"/v1/investigations/{investigation_id}/verify",
            json={
                "verification_status": "VERIFIED",
                "verified_cause": "CH",
            },
        ).status_code
        == 403
    )
    assert (
        client.patch(
            f"/v1/investigations/{investigation_id}",
            json={
                "investigation_status": "VERIFIED",
                "verification_status": "VERIFIED",
                "verified_cause": "CH",
            },
        ).status_code
        == 422
    )
    assert client.get(f"/v1/investigations/{investigation_id}").json()["verified_cause"] is None
    locked = client.patch(
        f"/v1/investigations/{investigation_id}", json={"findings": "late change"}
    )
    assert locked.status_code == 409
    client.__exit__(None, None, None)


def test_operator_cannot_modify_another_users_investigation(monkeypatch, tmp_path):
    client, database = setup_client(monkeypatch, tmp_path)
    _, investigation_id = create_investigation(client)
    client.app.state.auth.create_user("second", "second-password")
    client.post("/v1/auth/logout", json={})
    login = client.post(
        "/v1/auth/login", json={"username": "second", "password": "second-password"}
    )
    assert login.status_code == 200
    # A role header cannot change the authenticated database role or ownership.
    response = client.patch(
        f"/v1/investigations/{investigation_id}",
        json={"findings": "unauthorized"},
        headers={"X-Role": "network_admin"},
    )
    assert response.status_code == 404
    assert client.get(f"/v1/investigations/{investigation_id}").status_code == 404
    client.__exit__(None, None, None)


def test_network_admin_verifies_corrects_and_marks_unknown(monkeypatch, tmp_path):
    client, database = setup_client(monkeypatch, tmp_path)
    prediction_id, investigation_id = create_investigation(client)
    submit(client, investigation_id, "II")
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='network_admin'")
    invalid_normal = client.post(
        f"/v1/investigations/{investigation_id}/verify",
        json={
            "verification_status": "VERIFIED",
            "verified_cause": "Normal",
        },
    )
    assert invalid_normal.status_code == 422
    response = client.post(
        f"/v1/investigations/{investigation_id}/verify",
        json={
            "verification_status": "VERIFIED",
            "verified_cause": "II",
            "notes": "Evidence supports II.",
        },
    )
    assert response.status_code == 200, response.text
    forged_prediction = client.post(
        f"/v1/investigations/{investigation_id}/verify",
        json={"verification_status": "VERIFIED", "verified_cause": "EU", "predicted_cause": "II"},
    )
    assert forged_prediction.status_code == 422
    assert response.json()["predicted_cause"] == "CH"
    assert response.json()["investigation_cause"] == "II"
    assert response.json()["verified_cause"] == "II"
    assert response.json()["evaluation_eligible"] is True
    assert client.get(f"/v1/investigations/{investigation_id}/outcome").status_code == 200
    corrected = client.post(
        f"/v1/investigations/{investigation_id}/verify",
        json={
            "verification_status": "VERIFIED",
            "verified_cause": "EU",
            "notes": "Correction after review.",
        },
    )
    assert corrected.status_code == 200
    body = corrected.json()
    assert body["predicted_cause"] == "CH" and body["verified_cause"] == "EU"
    assert len(body["verification_history"]) == 2
    unresolved = client.post(
        f"/v1/investigations/{investigation_id}/verify",
        json={
            "verification_status": "UNKNOWN",
            "verified_cause": None,
            "notes": "Evidence insufficient.",
        },
    )
    assert unresolved.status_code == 200
    assert unresolved.json()["verified_cause"] is None
    assert unresolved.json()["evaluation_eligible"] is False
    assert unresolved.json()["predicted_cause"] == "CH"
    assert unresolved.json()["operational_context"]["kpis"]["retainability"] == 0.99
    assert unresolved.json()["operational_context"]["status"] == "FAULT"
    assert unresolved.json()["predicted_cause"] == "CH"
    queue = client.get("/v1/investigations/queue")
    assert queue.status_code == 200
    assert [item["investigation_id"] for item in queue.json()] == [investigation_id]
    assert queue.json()[0]["verification_status"] == "UNKNOWN"
    still_unresolved = client.post(
        f"/v1/investigations/{investigation_id}/verify",
        json={
            "verification_status": "UNRESOLVED",
            "verified_cause": None,
            "notes": "Follow-up could not establish a cause.",
        },
    )
    assert still_unresolved.status_code == 200
    assert still_unresolved.json()["verification_status"] == "UNRESOLVED"
    assert still_unresolved.json()["evaluation_eligible"] is False
    assert prediction_id == body["prediction_id"]
    client.__exit__(None, None, None)


def test_investigation_queue_is_network_admin_only_and_excludes_drafts(monkeypatch, tmp_path):
    client, _ = setup_client(monkeypatch, tmp_path)
    assert client.get("/v1/investigations/queue").status_code == 403
    client.app.state.auth.create_user("other-operator", "other-password")
    client.post("/v1/auth/logout", json={})
    login = client.post(
        "/v1/auth/login", json={"username": "other-operator", "password": "other-password"}
    )
    assert login.status_code == 200
    _, draft_id = create_investigation(client)
    client.post("/v1/auth/logout", json={})
    client.post("/v1/auth/login", json={"username": "operator", "password": "test-password"})
    _, submitted_id = create_investigation(client)
    submit(client, submitted_id)
    with sqlite3.connect(client.app.state.auth.path) as connection:
        connection.execute("UPDATE users SET role='network_admin' WHERE username='operator'")
    assert client.get(f"/v1/investigations/{draft_id}").status_code == 404
    response = client.get("/v1/investigations/queue")
    assert response.status_code == 200
    assert [item["investigation_id"] for item in response.json()] == [submitted_id]
    assert draft_id not in [item["investigation_id"] for item in response.json()]
    listed = client.get("/v1/investigations")
    assert listed.status_code == 200
    assert [item["investigation_id"] for item in listed.json()] == [submitted_id]
    client.__exit__(None, None, None)


def test_evaluable_outcomes_distinguish_prediction_finding_and_verified_cause(
    monkeypatch, tmp_path
):
    client, database = setup_client(monkeypatch, tmp_path)
    # CH predicted, CH finding, CH verified.
    _, same_id = create_investigation(client)
    submit(client, same_id, "CH")
    # CH predicted, II finding, II verified.
    _, different_id = create_investigation(client)
    submit(client, different_id, "II")
    # CH predicted, no established finding, unknown verified result.
    _, unknown_id = create_investigation(client)
    submit(client, unknown_id, None)
    # Open investigation remains ineligible.
    _, open_id = create_investigation(client)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='network_admin'")
    for item_id, cause in ((same_id, "CH"), (different_id, "II")):
        assert (
            client.post(
                f"/v1/investigations/{item_id}/verify",
                json={
                    "verification_status": "VERIFIED",
                    "verified_cause": cause,
                },
            ).status_code
            == 200
        )
    client.post(
        f"/v1/investigations/{unknown_id}/verify",
        json={
            "verification_status": "UNKNOWN",
            "verified_cause": None,
        },
    )
    assert client.get(f"/v1/investigations/{same_id}").json()["evaluation_eligible"] is True
    different = client.get(f"/v1/investigations/{different_id}").json()
    assert (
        different["predicted_cause"],
        different["investigation_cause"],
        different["verified_cause"],
    ) == ("CH", "II", "II")
    assert different["evaluation_eligible"] is True
    assert client.get(f"/v1/investigations/{unknown_id}").json()["evaluation_eligible"] is False
    assert client.get(f"/v1/investigations/{open_id}").json()["evaluation_eligible"] is False
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='platform_ml_admin'")
    eligible = client.get("/v1/investigations/evaluatable")
    assert eligible.status_code == 200
    assert {item["investigation_id"] for item in eligible.json()} == {same_id, different_id}
    client.__exit__(None, None, None)


def test_only_network_admin_can_verify_and_platform_ml_admin_is_read_only(monkeypatch, tmp_path):
    client, database = setup_client(monkeypatch, tmp_path)
    _, investigation_id = create_investigation(client)
    submit(client, investigation_id)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='network_admin'")
    verified = client.post(
        f"/v1/investigations/{investigation_id}/verify",
        json={"verification_status": "VERIFIED", "verified_cause": "CH"},
    )
    assert verified.status_code == 200
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='platform_ml_admin'")
    read_result = client.get(f"/v1/investigations/{investigation_id}/outcome")
    assert read_result.status_code == 200
    assert read_result.json()["verified_cause"] == "CH"
    assert read_result.json()["evaluation_eligible"] is True
    assert (
        client.post(
            f"/v1/investigations/{investigation_id}/verify",
            json={
                "verification_status": "VERIFIED",
                "verified_cause": "CH",
            },
        ).status_code
        == 403
    )
    evaluatable = client.get("/v1/investigations/evaluatable")
    assert {item["investigation_id"] for item in evaluatable.json()} == {investigation_id}
    client.__exit__(None, None, None)


def test_ml_evaluation_authorization_and_verified_cause_metrics(monkeypatch, tmp_path):
    client, database = setup_client(monkeypatch, tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='operator'")
    assert client.get("/v1/ml/evaluation").status_code == 403

    cases = [("CH", "CH"), ("CH", "II"), ("II", "II")]
    investigation_ids = []
    for index, (predicted, _verified) in enumerate(cases):
        prediction_id, investigation_id = create_investigation(client)
        investigation_ids.append(investigation_id)
        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE predictions SET cause=?,model_version=? WHERE id=?",
                (predicted, "test-model-v1" if index < 2 else "test-model-v2", prediction_id),
            )
            connection.execute(
                "UPDATE investigations SET predicted_cause=? WHERE id=?",
                (predicted, investigation_id),
            )
        submit(client, investigation_id)

    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='network_admin'")
    assert client.get("/v1/ml/evaluation").status_code == 403
    for investigation_id, (_, verified) in zip(investigation_ids, cases, strict=True):
        response = client.post(
            f"/v1/investigations/{investigation_id}/verify",
            json={"verification_status": "VERIFIED", "verified_cause": verified},
        )
        assert response.status_code == 200

    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='platform_ml_admin'")
        before = {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in ("predictions", "investigations", "investigation_verification_events")
        }
    result = client.get("/v1/ml/evaluation")
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["evaluation_scope"] == "verified_investigations"
    assert body["eligible_cases"] == 3
    assert body["cause_classification"] == {"accuracy": 2 / 3, "correct": 2, "incorrect": 1}
    assert body["by_cause"]["CH"] == {
        "support": 1,
        "predicted_count": 2,
        "correct": 1,
        "incorrect": 0,
        "precision": 0.5,
        "recall": 1.0,
        "f1": 2 / 3,
    }
    assert body["by_cause"]["II"]["support"] == 2
    assert body["confusion_matrix"]["class_order"] == ["ED", "CH", "II", "TLHO", "RP", "EU"]
    assert body["confusion_matrix"]["counts"]["II"]["CH"] == 1
    assert body["model_versions"] == [
        {
            "model_version": "test-model-v1",
            "eligible_cases": 2,
            "correct": 1,
            "incorrect": 1,
            "cause_classification_accuracy": 0.5,
        },
        {
            "model_version": "test-model-v2",
            "eligible_cases": 1,
            "correct": 1,
            "incorrect": 0,
            "cause_classification_accuracy": 1.0,
        },
    ]
    assert body["binary_detection"]["status"] == "not_yet_measurable"
    with sqlite3.connect(database) as connection:
        after = {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in before
        }
    assert after == before
    client.__exit__(None, None, None)


def test_ml_evaluation_excludes_nonverified_missing_and_invalid_outcomes(monkeypatch, tmp_path):
    client, database = setup_client(monkeypatch, tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='operator'")
    _, pending_id = create_investigation(client)
    submit(client, pending_id)
    _, unresolved_id = create_investigation(client)
    submit(client, unresolved_id)
    _, unknown_id = create_investigation(client)
    submit(client, unknown_id)
    _, eligible_id = create_investigation(client)
    submit(client, eligible_id)
    _, invalid_cause_id = create_investigation(client)
    submit(client, invalid_cause_id)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='network_admin'")
    for investigation_id, state in ((unresolved_id, "UNRESOLVED"), (unknown_id, "UNKNOWN")):
        response = client.post(
            f"/v1/investigations/{investigation_id}/verify",
            json={"verification_status": state, "verified_cause": None},
        )
        assert response.status_code == 200
    assert client.post(
        f"/v1/investigations/{eligible_id}/verify",
        json={"verification_status": "VERIFIED", "verified_cause": "CH"},
    ).status_code == 200
    assert client.post(
        f"/v1/investigations/{invalid_cause_id}/verify",
        json={"verification_status": "VERIFIED", "verified_cause": "CH"},
    ).status_code == 200
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE investigations SET verified_cause=NULL WHERE id=?", (eligible_id,)
        )
        connection.execute(
            "UPDATE investigations SET predicted_cause='INVALID' WHERE id=?",
            (invalid_cause_id,),
        )
        connection.execute("UPDATE users SET role='platform_ml_admin'")
    body = client.get("/v1/ml/evaluation").json()
    assert body["eligible_cases"] == 0
    assert body["cause_classification"] == {"accuracy": None, "correct": 0, "incorrect": 0}
    assert body["by_cause"] == {}
    assert body["excluded_cases"] == {
        "invalid_or_incomplete": 1,
        "missing_verified_cause": 1,
        "pending": 1,
        "unknown": 1,
        "unresolved": 1,
    }
    assert body["model_versions"] == []
    client.__exit__(None, None, None)


def test_ml_evaluation_aggregates_unknown_model_provenance_without_version_attribution(
    monkeypatch, tmp_path
):
    client, database = setup_client(monkeypatch, tmp_path)
    prediction_id, investigation_id = create_investigation(client)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE predictions SET model_version='legacy-unknown' WHERE id=?",
            (prediction_id,),
        )
    submit(client, investigation_id, "CH")
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='network_admin'")
    verified = client.post(
        f"/v1/investigations/{investigation_id}/verify",
        json={"verification_status": "VERIFIED", "verified_cause": "CH"},
    )
    assert verified.status_code == 200
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='platform_ml_admin'")

    body = client.get("/v1/ml/evaluation").json()
    assert body["eligible_cases"] == 1
    assert body["cause_classification"] == {"accuracy": 1.0, "correct": 1, "incorrect": 0}
    assert body["model_versions"] == []
    client.__exit__(None, None, None)


def test_ml_evaluation_single_class_and_network_health_version(monkeypatch, tmp_path):
    client, database = setup_client(monkeypatch, tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='operator'")
    run_id = 1
    with sqlite3.connect(database) as connection:
        cursor = connection.execute(
            "INSERT INTO network_health_runs(dataset_name,status,started_at,model_version) "
            "VALUES('test.csv','COMPLETED','2026-10-01T00:00:00+00:00','nh-model-v1')"
        )
        run_id = cursor.lastrowid
        cursor = connection.execute(
            """INSERT INTO network_health_results
            (run_id,cell,latitude,longitude,ground_truth_cause,status,predicted_cause,
             model_version,result_json) VALUES(?,?,?,?,?,?,?,?,?)""",
            (run_id, "CELL-EVAL", 1.0, 1.0, 1, "FAULT", "ED", "nh-model-v1", "{}"),
        )
        result_id = cursor.lastrowid
    created = client.post("/v1/investigations", json={"network_health_result_id": result_id})
    assert created.status_code == 201
    investigation_id = created.json()["investigation_id"]
    submit(client, investigation_id, "ED")
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='network_admin'")
    assert client.post(
        f"/v1/investigations/{investigation_id}/verify",
        json={"verification_status": "VERIFIED", "verified_cause": "ED"},
    ).status_code == 200
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='platform_ml_admin'")
    body = client.get("/v1/ml/evaluation").json()
    assert body["eligible_cases"] == 1
    assert body["cause_classification"]["accuracy"] == 1.0
    assert set(body["by_cause"]) == {"ED"}
    assert body["by_cause"]["ED"]["f1"] == 1.0
    assert body["model_versions"][0]["model_version"] == "nh-model-v1"
    client.__exit__(None, None, None)


def test_verified_outcomes_are_read_only_admin_evidence_with_shared_eligibility(
    monkeypatch, tmp_path
):
    client, database = setup_client(monkeypatch, tmp_path)
    empty = client.get("/v1/ml/verified-outcomes")
    assert empty.status_code == 403
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='platform_ml_admin'")
    empty = client.get("/v1/ml/verified-outcomes")
    assert empty.status_code == 200
    assert empty.json()["eligible_cases"] == 0
    assert empty.json()["items"] == []
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='operator'")

    match_prediction, match_id = create_investigation(client)
    submit(client, match_id, "CH")
    mismatch_prediction, mismatch_id = create_investigation(client)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE predictions SET cause='ED' WHERE id=?", (mismatch_prediction,))
        connection.execute(
            "UPDATE investigations SET predicted_cause='ED' WHERE id=?", (mismatch_id,)
        )
    submit(client, mismatch_id, "ED")
    _, unresolved_id = create_investigation(client)
    submit(client, unresolved_id, "CH")
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='network_admin'")
    for case_id, status, cause in (
        (match_id, "VERIFIED", "CH"),
        (mismatch_id, "VERIFIED", "CH"),
        (unresolved_id, "UNRESOLVED", None),
    ):
        assert client.post(
            f"/v1/investigations/{case_id}/verify",
            json={"verification_status": status, "verified_cause": cause},
        ).status_code == 200
    assert client.get("/v1/ml/verified-outcomes").status_code == 403
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='operator'")
    assert client.get("/v1/ml/verified-outcomes").status_code == 403
    client.cookies.clear()
    assert client.get("/v1/ml/verified-outcomes").status_code == 401
    assert client.post(
        "/v1/auth/login", json={"username": "operator", "password": "test-password"}
    ).status_code == 200
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role='platform_ml_admin'")
        connection.execute(
            "UPDATE predictions SET model_version='legacy-unknown' WHERE id=?",
            (mismatch_prediction,),
        )
        before = connection.execute(
            "SELECT id,cause,model_version FROM predictions ORDER BY id"
        ).fetchall()
    result = client.get("/v1/ml/verified-outcomes")
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["verified_cases"] == 2
    assert body["eligible_cases"] == client.get("/v1/ml/evaluation").json()["eligible_cases"] == 2
    assert body["excluded_cases"]["unresolved"] == 1
    cases = {item["investigation_id"]: item for item in body["items"]}
    assert set(cases) == {match_id, mismatch_id}
    assert cases[match_id]["evaluation_match"] is True
    assert cases[match_id]["model_version"] == "test-model"
    assert cases[mismatch_id]["predicted_cause"] == "ED"
    assert cases[mismatch_id]["verified_cause"] == "CH"
    assert cases[mismatch_id]["evaluation_match"] is False
    assert cases[mismatch_id]["model_version"] is None
    assert "findings" in cases[match_id] and "kpis" not in cases[match_id]
    with sqlite3.connect(database) as connection:
        after = connection.execute(
            "SELECT id,cause,model_version FROM predictions ORDER BY id"
        ).fetchall()
    assert after == before
    client.__exit__(None, None, None)


def test_network_health_result_identifier_can_be_investigated_without_copying_prediction(
    monkeypatch, tmp_path
):
    client, database = setup_client(monkeypatch, tmp_path)
    with sqlite3.connect(database) as connection:
        cursor = connection.execute(
            "INSERT INTO network_health_runs(dataset_name,status,started_at) "
            "VALUES('test.csv','COMPLETED','2026-10-01T00:00:00+00:00')"
        )
        run_id = cursor.lastrowid
        cursor = connection.execute(
            """INSERT INTO network_health_results
            (run_id,cell,latitude,longitude,ground_truth_cause,status,predicted_cause,model_version,result_json)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (run_id, "CELL-NH", 36.6, 3.1, 2, "FAULT", "CH", "test-model", "{}"),
        )
        result_id = cursor.lastrowid
    page = client.get("/v1/network-health/results?limit=1")
    assert page.status_code == 200
    assert page.json()["items"][0]["network_health_result_id"] == result_id
    created = client.post("/v1/investigations", json={"network_health_result_id": result_id})
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["network_health_result_id"] == result_id
    assert body["prediction_id"] is None
    assert body["predicted_cause"] == "CH"
    assert body["verified_cause"] is None
    assert body["evaluation_eligible"] is False
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
    client.__exit__(None, None, None)


def test_verified_normal_is_rejected_and_migration_downgrades_cleanly(tmp_path):
    from alembic import command
    from alembic.config import Config

    database = tmp_path / "investigation-migration.sqlite3"
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database.resolve().as_posix()}")
    command.upgrade(config, "0005_user_roles")
    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name='investigations'"
        ).fetchone()
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name='investigation_verification_events'"
        ).fetchone()
    command.downgrade(config, "0005_user_roles")
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='investigations'"
            ).fetchone()
            is None
        )
    assert FaultCause.NORMAL.value == "Normal"
