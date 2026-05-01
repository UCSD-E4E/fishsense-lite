"""Dynaconf settings for the backup worker.

Mirrors the same E4EFS_ envvar prefix + dynaconf-validators-up-front
pattern the other workers use. Adds a `postgres.*` section (this is
the only worker that needs DB access) and a `backup.*` section for
the schedule + retention shape.
"""

import logging
from importlib.metadata import version

import validators
from dynaconf import Dynaconf, Validator

from fishsense_shared import (
    configure_logging as _configure_logging,
    get_config_path,
    path_validator,
    url_condition,
)

APP_NAME = "e4efs_backup_worker"


_VALIDATORS = [
    Validator(
        "general.max_workers",
        required=True,
        cast=int,
        default=2,
        condition=lambda x: x > 0,
    ),
    # Temporal cluster (shared with the rest of the workers).
    Validator("temporal.host", required=True, cast=str, condition=validators.hostname),
    Validator("temporal.port", required=True, cast=int, default=7233),
    Validator("temporal.tls", required=True, cast=bool, default=False),
    Validator("temporal.client_cert", cast=str, condition=path_validator),
    Validator("temporal.client_private_key", cast=str, condition=path_validator),
    Validator("temporal.domain", cast=str),
    Validator("temporal.server_root_ca_cert", cast=str, condition=path_validator),
    # NAS (where dumps land).
    Validator("e4e_nas.url", required=True, cast=str, condition=url_condition),
    Validator("e4e_nas.username", required=True, cast=str),
    Validator("e4e_nas.password", required=True, cast=str),
    # Postgres (this worker is the ONLY one with DB credentials — keep
    # the role narrow).
    Validator("postgres.host", required=True, cast=str, condition=validators.hostname),
    Validator("postgres.port", required=True, cast=int, default=5432),
    Validator("postgres.username", required=True, cast=str),
    Validator("postgres.password", required=True, cast=str),
    # Backup shape. databases is a list of PG database names; the order
    # is preserved in the workflow's fanout but doesn't otherwise matter.
    Validator(
        "backup.databases",
        required=True,
        default=["fishsense", "superset", "temporal_db"],
    ),
    Validator(
        "backup.retention_count",
        required=True,
        cast=int,
        default=14,
        condition=lambda x: x > 0,
    ),
    Validator(
        "backup.nas_root_path",
        required=True,
        cast=str,
        default="/fishsense_backups",
    ),
    Validator(
        "backup.schedule_id",
        required=True,
        cast=str,
        default="fishsense-daily-db-backup",
    ),
    # Standard 5-field cron. Default: daily at 03:00 UTC.
    Validator(
        "backup.schedule_cron",
        required=True,
        cast=str,
        default="0 3 * * *",
    ),
    Validator(
        "backup.task_queue",
        required=True,
        cast=str,
        default="fishsense_backup_queue",
    ),
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


def configure_logging() -> None:
    """Configure logging for this service and emit the version banner."""
    _configure_logging(APP_NAME, log_filename=f"{APP_NAME}.log")
    logging.info(
        "Executing fishsense-backup-worker:%s",
        version("fishsense-backup-worker"),
    )
