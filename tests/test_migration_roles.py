import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config

from app.auth import PASSWORD_HASHER, AuthRepository, Role


def alembic_config(database: Path) -> Config:
    config = Config(str(Path(__file__).parents[1] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database.resolve().as_posix()}")
    return config


def test_role_migration_preserves_existing_account_and_defaults_to_operator(tmp_path: Path):
    database = tmp_path / "pre-rbac.sqlite3"
    config = alembic_config(database)
    command.upgrade(config, "0004_network_health")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO users(username,password_hash,is_active,created_at) VALUES(?,?,?,?)",
            ("existing-user", PASSWORD_HASHER.hash("password"), 1, "2026-01-01T00:00:00+00:00"),
        )

    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT username, role FROM users").fetchone()
    assert row == ("existing-user", "operator")
    existing_user = AuthRepository(database, 3600).authenticate("existing-user", "password")
    assert existing_user is not None and existing_user.role is Role.OPERATOR

    with sqlite3.connect(database) as connection:
        try:
            connection.execute(
                "INSERT INTO users(username,password_hash,is_active,created_at,role) "
                "VALUES(?,?,?,?,?)",
                ("invalid-role", "hash", 1, "2026-01-01T00:00:00+00:00", "root"),
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("database accepted an unsupported role")

    command.downgrade(config, "0004_network_health")
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT username FROM users").fetchone() == ("existing-user",)
