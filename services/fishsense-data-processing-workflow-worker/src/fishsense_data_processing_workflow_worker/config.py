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

APP_NAME = "e4efs_data_processing_workflow_worker"


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
    # Which Temporal namespace to connect to. OSS mTLS doesn't pin the client
    # to one (krg-infra ADR 0023), so it MUST be requested explicitly; prod
    # settings.toml sets `fishsense`. Defaults to `default` for local dev/tests.
    Validator("temporal.namespace", cast=str, default="default"),
    Validator("temporal.server_root_ca_cert", cast=str, condition=path_validator),
    Validator("fishsense_api.url", required=True, cast=str, condition=url_condition),
    Validator("fishsense_api.username", cast=str),
    Validator("fishsense_api.password", cast=str),
    # Garage (S3-compatible) object store — replaces the nginx
    # file-exchange. The data-worker reads staged raw `.ORF` + slate
    # PDFs and writes processed JPEGs back. S3 access keys work from any
    # IP (no IP allowlist / forward-auth needed), which is what lets the
    # data-worker run off-prem (NRP) without a stable egress IP.
    # `access_key`/`secret_key` live in `.secrets.toml`.
    Validator(
        "object_store.endpoint_url",
        required=True,
        cast=str,
        condition=url_condition,
    ),
    Validator("object_store.region", required=True, cast=str, default="garage"),
    # `bucket` = raw/slate **scratch** (read here). Processed JPEGs are
    # written to `labels_bucket` under `labels_prefix` — the LS-facing bucket.
    # `labels_bucket` defaults to `bucket`; `labels_prefix` defaults to "".
    Validator("object_store.bucket", required=True, cast=str),
    Validator("object_store.labels_bucket", cast=str),
    Validator("object_store.labels_prefix", cast=str, default=""),
    Validator("object_store.access_key", required=True, cast=str),
    Validator("object_store.secret_key", required=True, cast=str),
]

# NOTE: standardized on E4EFS_ envvar prefix (was DYNACONF_) to match the other
# services in this monorepo. Existing DYNACONF_* env vars must be renamed before
# deploying this version of the worker.
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
        "Executing fishsense-data-processing-workflow-worker:%s",
        version("fishsense-data-processing-workflow-worker"),
    )
