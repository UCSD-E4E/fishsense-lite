"""Path conventions shared by every FishSense service.

When ``E4EFS_DOCKER=true`` (set on every shipped image), config and logs are
read from / written to ``/e4efs/{config,logs}`` instead of the platform user
dirs. The ``app_name`` argument distinguishes log destinations between
services co-installed on a host (e.g. via platformdirs).
"""

import os
from pathlib import Path
from urllib.parse import urlparse

import platformdirs

# True only when E4EFS_DOCKER is an explicitly-truthy value. NOT
# `bool(os.environ.get("E4EFS_DOCKER"))` — that treats *any* non-empty
# string as true, so `E4EFS_DOCKER=false` would (wrongly) read as
# Docker mode, sending config/log paths to `/e4efs/*` even where those
# don't exist (e.g. the local devcontainer, which sets it to "false").
IS_DOCKER = os.environ.get("E4EFS_DOCKER", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def get_config_path() -> Path:
    """Config root: ``/e4efs/config`` in Docker, else cwd."""
    if IS_DOCKER:
        return Path("/e4efs/config")
    return Path(".")


def get_log_path(app_name: str) -> Path:
    """Log root: ``/e4efs/logs`` in Docker, else platformdirs user log path."""
    if IS_DOCKER:
        return Path("/e4efs/logs")
    log_path = platformdirs.PlatformDirs(app_name).user_log_path
    log_path.mkdir(parents=True, exist_ok=True)
    return log_path


def path_validator(path: str) -> bool:
    """Dynaconf validator: True when ``path`` exists on disk."""
    return Path(path).exists()


def url_condition(value: str) -> bool:
    """Dynaconf validator: permissive http/https URL with a hostname.

    `validators.url` rejects Docker-internal hostnames (underscores, no
    TLD) like `http://static_file_server` and `http://fishsense-api:8000`,
    which is what the local devcontainer + production compose stacks
    actually use. This still catches typos like a missing scheme.
    """
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)
