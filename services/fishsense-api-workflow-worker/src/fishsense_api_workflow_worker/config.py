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
    # --- Kubernetes scale-to-zero for the NRP data-worker ---
    # The api-worker drives the data-processing worker's replica count:
    # parent workflows scale it up to `active_replicas` before
    # dispatching a child, and an hourly sweeper scales it back to 0
    # when no data-worker workflows are running. Scaling is OFF unless
    # `kubernetes.kubeconfig_path` points at a readable NRP kubeconfig
    # — without it the worker is assumed always-on (the pre-NRP
    # behavior), so the local devcontainer and tests don't need any of
    # these. When `kubeconfig_path` IS set, `kubernetes.namespace` is
    # required (the scaling activity raises if it's missing).
    Validator("kubernetes.kubeconfig_path", cast=str, condition=path_validator),
    Validator("kubernetes.namespace", cast=str),
    Validator(
        "kubernetes.deployment_name",
        cast=str,
        default="fishsense-data-processing-workflow-worker",
    ),
    # Hard-capped at 4: >1 is only ever a deliberate operator choice
    # (a giant single dive, or active-window resilience on a
    # preemption-prone cluster); the scaling activity clamps to
    # [1, 4] so a fat-fingered value can't ask NRP for 50 pods.
    Validator(
        "kubernetes.active_replicas",
        cast=int,
        default=1,
        condition=lambda x: 1 <= x <= 4,
    ),
    # The sweeper refuses to scale to 0 until the data-worker task
    # queue has had no running OR recently-closed workflow for this
    # many minutes — so a back-to-back dive doesn't thrash the pod.
    Validator(
        "kubernetes.idle_cooldown_minutes",
        cast=int,
        default=15,
        condition=lambda x: x >= 0,
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
        "Executing fishsense_api_workflow_worker:%s",
        version("fishsense_api_workflow_worker"),
    )
