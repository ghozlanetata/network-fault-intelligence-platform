import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.auth import AuthRepository, Role
from app.config import get_settings


@pytest.fixture
def client(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "auth.sqlite3"
    monkeypatch.setenv("NFI_DATABASE_PATH", str(db_path))
    monkeypatch.setenv("NFI_BOOTSTRAP_USERNAME", "operator@example.test")
    monkeypatch.setenv("NFI_BOOTSTRAP_PASSWORD", "correct-horse-battery")
    monkeypatch.setenv("NFI_BOOTSTRAP_ROLE", "operator")
    monkeypatch.setenv("NFI_MODEL_DIR", str(tmp_path / "missing-models"))
    monkeypatch.setenv("NFI_SESSION_TTL", "60")
    get_settings.cache_clear()
    with TestClient(main_module.app) as test_client:
        yield test_client, db_path
    get_settings.cache_clear()


def login(client: TestClient):
    return client.post(
        "/v1/auth/login",
        json={"username": "operator@example.test", "password": "correct-horse-battery"},
    )


def test_bootstrap_hashes_password_and_does_not_overwrite(tmp_path: Path):
    repository = AuthRepository(tmp_path / "users.sqlite3", 60)
    from app.repository import HistoryRepository

    HistoryRepository(repository.path).initialize()
    repository.bootstrap("operator", "one-password")
    repository.bootstrap("operator", "replacement-password")
    with sqlite3.connect(repository.path) as connection:
        saved = connection.execute("SELECT password_hash FROM users").fetchone()[0]
    assert saved != "one-password"
    assert "one-password" not in saved
    assert repository.authenticate("operator", "one-password") is not None
    assert repository.authenticate("operator", "replacement-password") is None


@pytest.mark.parametrize("credentials", [("absent", "wrong"), ("operator@example.test", "wrong")])
def test_failed_login_has_generic_response(client, credentials):
    test_client, _ = client
    response = test_client.post(
        "/v1/auth/login", json={"username": credentials[0], "password": credentials[1]}
    )
    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Username or password is incorrect."


def test_session_me_cookie_logout_and_protected_endpoints(client):
    test_client, _ = client
    assert test_client.get("/v1/auth/me").status_code == 401
    response = login(test_client)
    assert response.status_code == 200
    assert response.json() == {
        "id": 1,
        "username": "operator@example.test",
        "is_active": True,
        "role": "operator",
    }
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie
    assert "correct-horse-battery" not in response.text
    assert test_client.get("/v1/auth/me").status_code == 200
    assert test_client.get("/v1/history").status_code == 200
    assert test_client.post("/v1/predictions", json={}).status_code == 422
    assert test_client.post("/v1/auth/logout", json={}).status_code == 204
    assert test_client.get("/v1/auth/me").status_code == 401
    assert test_client.get("/v1/history").status_code == 401
    assert test_client.post("/v1/predictions", json={}).status_code == 401


def test_unknown_session_and_expired_session_are_rejected(client):
    test_client, database = client
    test_client.cookies.set("nfi_session", "invalid-token")
    assert test_client.get("/v1/auth/me").status_code == 401
    test_client.cookies.clear()
    response = login(test_client)
    raw_cookie = response.cookies["nfi_session"]
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE sessions SET expires_at = ?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(),),
        )
    assert raw_cookie
    assert test_client.get("/v1/auth/me").status_code == 401


def test_inactive_user_cannot_authenticate_or_use_session(client):
    test_client, database = client
    assert login(test_client).status_code == 200
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET is_active = 0")
    response = test_client.post(
        "/v1/auth/login",
        json={"username": "operator@example.test", "password": "correct-horse-battery"},
    )
    assert response.status_code == 401
    assert test_client.get("/v1/auth/me").status_code == 401


def test_health_endpoints_remain_public(client):
    test_client, _ = client
    assert test_client.get("/health/live").status_code == 200
    assert test_client.get("/health/ready").status_code == 200
    assert test_client.get("/docs").status_code == 401


def test_cross_origin_login_is_rejected(client):
    test_client, _ = client
    response = test_client.post(
        "/v1/auth/login",
        json={"username": "operator@example.test", "password": "correct-horse-battery"},
        headers={"origin": "https://attacker.invalid"},
    )
    assert response.status_code == 403


def test_database_role_is_returned_and_changes_apply_to_existing_session(client):
    test_client, database = client
    assert login(test_client).json()["role"] == "operator"
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role = ?", (Role.NETWORK_ADMIN.value,))
    assert test_client.post(
        "/v1/auth/login",
        json={"username": "operator@example.test", "password": "correct-horse-battery"},
    ).json()["role"] == "network_admin"
    assert test_client.get("/v1/auth/me").json()["role"] == "network_admin"
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE users SET role = ?", (Role.PLATFORM_ML_ADMIN.value,))
    assert test_client.post(
        "/v1/auth/login",
        json={"username": "operator@example.test", "password": "correct-horse-battery"},
    ).json()["role"] == "platform_ml_admin"
    assert test_client.get("/v1/auth/me").json()["role"] == "platform_ml_admin"


def test_role_cannot_be_set_by_login_payload_or_arbitrary_repository_value(client, tmp_path):
    test_client, _ = client
    response = test_client.post(
        "/v1/auth/login",
        json={
            "username": "operator@example.test",
            "password": "correct-horse-battery",
            "role": "platform_ml_admin",
        },
    )
    assert response.status_code == 422
    assert login(test_client).json()["role"] == "operator"

    repository = AuthRepository(tmp_path / "invalid-role.sqlite3", 60)
    from app.repository import HistoryRepository

    HistoryRepository(repository.path).initialize()
    with pytest.raises(ValueError):
        repository.create_user("invalid", "password", role="root")


def test_bootstrap_privileged_role_is_server_selected_and_does_not_promote_existing_user(
    tmp_path: Path,
):
    repository = AuthRepository(tmp_path / "bootstrap-role.sqlite3", 60)
    from app.repository import HistoryRepository

    HistoryRepository(repository.path).initialize()
    repository.bootstrap("first-admin", "secret", Role.PLATFORM_ML_ADMIN)
    assert repository.authenticate("first-admin", "secret").role is Role.PLATFORM_ML_ADMIN
    repository.bootstrap("first-admin", "new-secret", Role.NETWORK_ADMIN)
    assert repository.authenticate("first-admin", "secret").role is Role.PLATFORM_ML_ADMIN
    assert repository.authenticate("first-admin", "new-secret") is None
