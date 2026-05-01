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
    # URL prefix embedded into Label Studio task `data.image` fields —
    # labelers' browsers fetch from here through Traefik/authentik.
    # Public-facing; the file-exchange (`/api/v1/exchange/*`) is NOT
    # routed publicly, so this URL works only for `/api/v1/data/*`.
    Validator(
        "label_studio.image_url_base",
        required=True,
        cast=str,
        condition=url_condition,
    ),
    # Internal docker URL for the static_file_server nginx that brokers
    # the worker file-exchange (raw ORFs, slate PDFs, processed JPEGs).
    # Same backend as `label_studio.image_url_base` but on the docker
    # network, bypassing Traefik/authentik. Stage 12 fetches slate PDFs
    # from here to compute the composite-image PDF panel offset.
    Validator(
        "file_exchange.url",
        required=True,
        cast=str,
        condition=url_condition,
    ),
    Validator("e4e_nas.url", required=True, cast=str, condition=url_condition),
    Validator("e4e_nas.username", required=True, cast=str),
    Validator("e4e_nas.password", required=True, cast=str),
    # NAS path under which Phase 3b's archive activity writes
    # processed JPEGs. Per-stage subfolders + per-dive subfolders
    # are appended at archive time; the final NAS path is
    # `{processed_jpegs.nas_root_path}/{workflow}/{dive_id}/{checksum}.JPG`.
    Validator(
        "processed_jpegs.nas_root_path",
        required=True,
        cast=str,
        default="/fishsense_process_work/processed_jpegs",
    ),
    Validator("fishsense_api.url", required=True, cast=str, condition=url_condition),
    Validator("fishsense_api.username", cast=str),
    Validator("fishsense_api.password", cast=str),
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
