"""Dynaconf settings module."""

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

APP_NAME = "e4efs_api_workflow_worker"

_VALIDATORS = [
    Validator(
        "general.max_workers",
        required=True,
        cast=int,
        default=4,
        condition=lambda x: x > 0,
    ),
    Validator("temporal.host", required=True, cast=str, condition=validators.hostname),
    Validator("temporal.port", required=True, cast=int, default=7233),
    Validator("temporal.tls", required=True, cast=bool, default=False),
    Validator("temporal.client_cert", cast=str, condition=path_validator),
    Validator("temporal.client_private_key", cast=str, condition=path_validator),
    Validator("temporal.domain", cast=str),
    Validator("temporal.server_root_ca_cert", cast=str, condition=path_validator),
    Validator("label_studio.url", required=True, condition=url_condition),
    Validator("label_studio.api_key", required=True, cast=str),
    Validator("e4e_nas.url", required=True, cast=str, condition=url_condition),
    Validator("e4e_nas.username", required=True, cast=str),
    Validator("e4e_nas.password", required=True, cast=str),
    # NAS path prefix prepended to relative `image.path` / `dive_slate.path`
    # values stored in the DB before downloading from FileStation. The DB
    # stores paths relative to the lab's data-root share (e.g.
    # `2024.06.20.REEF/08_2023/.../P8290052.ORF`); the actual NAS location
    # is `/fishsense_data/REEF/data/<that>`. Without this prefix, every
    # `stage_raw_bytes_for_dive_activity` call lands at a path FileStation
    # can't resolve and fails with a 502 (Synology's WebAPI surfaces the
    # missing-path as Bad Gateway on the download endpoint specifically).
    Validator(
        "e4e_nas.raw_root_path",
        required=True,
        cast=str,
        default="/fishsense_data/REEF/data",
    ),
    Validator("fishsense_api.url", required=True, cast=str, condition=url_condition),
    Validator("fishsense_api.username", cast=str),
    Validator("fishsense_api.password", cast=str),
    # Garage (S3-compatible) object store — replaces the nginx
    # file-exchange. Single bucket; the data-worker reads staged raw
    # `.ORF` + slate PDFs from it and writes processed JPEGs back. This
    # worker stages raw/slate in and cleans up the `raw/` scratch
    # prefix. `access_key`/`secret_key` live in `.secrets.toml`.
    Validator(
        "object_store.endpoint_url",
        required=True,
        cast=str,
        condition=url_condition,
    ),
    Validator("object_store.region", required=True, cast=str, default="garage"),
    Validator("object_store.bucket", required=True, cast=str),
    Validator("object_store.access_key", required=True, cast=str),
    Validator("object_store.secret_key", required=True, cast=str),
    # Optional read-only key handed to Label Studio when registering the
    # per-dive S3 source storage so LS can presign GET URLs for the
    # processed JPEGs. Falls back to `access_key`/`secret_key` when
    # unset — ops can scope a read-only key here without code changes.
    Validator("object_store.presign_access_key", cast=str),
    Validator("object_store.presign_secret_key", cast=str),
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
        "Executing fishsense_api_workflow_worker:%s",
        version("fishsense_api_workflow_worker"),
    )
