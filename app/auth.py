"""Password and server-side session services for the same-origin operator UI."""

import hashlib
import secrets
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from pwdlib import PasswordHash

PASSWORD_HASHER = PasswordHash.recommended()


class Role(StrEnum):
    OPERATOR = "operator"
    NETWORK_ADMIN = "network_admin"
    PLATFORM_ML_ADMIN = "platform_ml_admin"


@dataclass(frozen=True)
class User:
    id: int
    username: str
    is_active: bool
    role: Role


class AuthRepository:
    def __init__(self, path: Path, session_ttl: int) -> None:
        self.path = path
        self.session_ttl = session_ttl

    def create_user(
        self,
        username: str,
        password: str,
        *,
        active: bool = True,
        role: Role = Role.OPERATOR,
    ) -> bool:
        username = normalize_username(username)
        if not username or not password:
            raise ValueError("Username and password are required.")
        role = Role(role)
        password_hash = PASSWORD_HASHER.hash(password)
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now(),))
            try:
                connection.execute(
                    "INSERT INTO users (username, password_hash, is_active, created_at, role) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (username, password_hash, int(active), now(), role.value),
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def bootstrap(
        self,
        username: str | None,
        password: str | None,
        role: Role = Role.OPERATOR,
    ) -> None:
        if not username and not password:
            return
        if not username or not password:
            raise ValueError("Both NFI_BOOTSTRAP_USERNAME and NFI_BOOTSTRAP_PASSWORD are required.")
        # The unique constraint makes concurrent startup safe and never overwrites an account.
        self.create_user(username, password, role=role)

    def authenticate(self, username: str, password: str) -> User | None:
        normalized = normalize_username(username)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT id, username, password_hash, is_active, role FROM users "
                "WHERE username = ?",
                (normalized,),
            ).fetchone()
        if row is None:
            # Keep unknown-user and wrong-password paths equivalent in observable response.
            PASSWORD_HASHER.verify(password, _DUMMY_HASH)
            return None
        try:
            valid = PASSWORD_HASHER.verify(password, row[2])
        except (ValueError, TypeError):
            valid = False
        if not valid or not row[3]:
            return None
        return User(int(row[0]), row[1], bool(row[3]), Role(row[4]))

    def create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        created = datetime.now(UTC)
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (created.isoformat(),))
            connection.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    token_hash(token),
                    user_id,
                    created.isoformat(),
                    (created + timedelta(seconds=self.session_ttl)).isoformat(),
                ),
            )
        return token

    def user_for_session(self, token: str | None) -> User | None:
        if not token:
            return None
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT users.id, users.username, users.is_active, sessions.expires_at, users.role "
                "FROM sessions JOIN users ON users.id = sessions.user_id "
                "WHERE sessions.token_hash = ?",
                (token_hash(token),),
            ).fetchone()
            if row is None:
                return None
            if row[3] <= now() or not row[2]:
                connection.execute(
                    "DELETE FROM sessions WHERE token_hash = ?", (token_hash(token),)
                )
                return None
        return User(int(row[0]), row[1], bool(row[2]), Role(row[4]))

    def delete_session(self, token: str | None) -> None:
        if token:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    "DELETE FROM sessions WHERE token_hash = ?", (token_hash(token),)
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def normalize_username(username: str) -> str:
    return username.strip().casefold()


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


# Used only to equalize unknown-user password verification cost.
_DUMMY_HASH = PASSWORD_HASHER.hash("not-a-real-password")
