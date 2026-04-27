"""Configuration module for FishSense API."""

import validators
from dynaconf import Dynaconf, Validator

from fishsense_shared import get_config_path

_VALIDATORS = [
    Validator("postgres.host", required=True, cast=str, condition=validators.hostname),
    Validator("postgres.port", required=True, cast=int, default=5432),
    Validator("postgres.username", required=True, cast=str),
    Validator("postgres.password", required=True, cast=str),
]

settings = Dynaconf(
    envvar_prefix="E4EFS",
    environments=False,
    settings_files=[
        (get_config_path() / "settings.toml").as_posix(),
        (get_config_path() / ".secrets.toml").as_posix(),
    ],
    merge_enabled=True,
    validators=_VALIDATORS,
)

def pg_connection_string() -> str:
    """Build the PostgreSQL connection string from current settings."""
    return (
        f"postgresql+asyncpg://{settings.postgres.username}:{settings.postgres.password}"
        f"@{settings.postgres.host}:{settings.postgres.port}/{settings.postgres.database}"
    )
