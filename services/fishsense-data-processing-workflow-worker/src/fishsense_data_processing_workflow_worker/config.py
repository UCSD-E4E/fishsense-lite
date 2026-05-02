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
    Validator("temporal.server_root_ca_cert", cast=str, condition=path_validator),
    Validator("fishsense_api.url", required=True, cast=str, condition=url_condition),
    Validator("fishsense_api.username", cast=str),
    Validator("fishsense_api.password", cast=str),
    # Internal docker URL for the static_file_server nginx that brokers
    # the worker file-exchange (raw ORFs, slate PDFs, processed JPEGs).
    Validator(
        "file_exchange.url", required=True, cast=str, condition=url_condition
    ),
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
