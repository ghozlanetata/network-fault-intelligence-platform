from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.auth import Role


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NFI_", env_file=".env", extra="ignore")

    database_path: Path = Path("./data/nfi.sqlite3")
    model_dir: Path = Path("./models")
    log_level: str = "INFO"
    session_ttl: int = Field(default=28800, gt=0)
    cookie_secure: bool = False
    cookie_samesite: Literal["lax", "strict", "none"] = "lax"
    bootstrap_username: str | None = None
    bootstrap_password: str | None = None
    bootstrap_role: Role = Role.OPERATOR


@lru_cache
def get_settings() -> Settings:
    return Settings()
