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
    # Which Temporal namespace to connect to. OSS mTLS doesn't pin the client
    # to one (krg-infra ADR 0023), so it MUST be requested explicitly; prod
    # settings.toml sets `fishsense`. Defaults to `default` for local dev/tests.
    Validator("temporal.namespace", cast=str, default="default"),
    Validator("temporal.server_root_ca_cert", cast=str, condition=path_validator),
    Validator("label_studio.url", required=True, condition=url_condition),
    Validator("label_studio.api_key", required=True, cast=str),
    # LS Enterprise workspace new per-dive projects are created in. Empty =
    # personal/default workspace (back-compat for OSS LS / local dev / tests).
    Validator("label_studio.workspace", cast=str, default=""),
    Validator("e4e_nas.url", required=True, cast=str, condition=url_condition),
    Validator("e4e_nas.username", required=True, cast=str),
    Validator("e4e_nas.password", required=True, cast=str),
    # Max concurrent raw `.ORF` downloads per staging activity. FileStation's
    # download backend (DSM nginx -> synoscgi) is a fragile shared CGI that
    # 502s / falls over under concurrent large transfers, so default to a
    # single serial stream. Tunable via config (E4EFS_E4E_NAS__STAGE_CONCURRENCY)
    # so we can ramp it up 1 -> 2 -> 3 while watching the NAS, without a redeploy.
    Validator("e4e_nas.stage_concurrency", cast=int, default=1),
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
    # >1 is only ever a deliberate operator choice (a giant single
    # dive, or active-window resilience on a preemption-prone cluster).
    # No `condition=` here: `resolve_scaling_config` clamps the value to
    # [1, 4], so a fat-fingered value is clamped (with the actual count
    # logged by the scaling activity), not a worker-startup error.
    Validator("kubernetes.active_replicas", cast=int, default=1),
    # The sweeper refuses to scale to 0 until the data-worker task
    # queue has had no running OR recently-closed workflow for this
    # many minutes — so a back-to-back dive doesn't thrash the pod.
    # `resolve_scaling_config` floors a negative value at 0.
    Validator("kubernetes.idle_cooldown_minutes", cast=int, default=15),
    # --- light queue ---------------------------------------------------------
    # `fishsense_data_processing_light_queue` is served by its own Deployment,
    # running the same image with `E4EFS_GENERAL__ROLE=light`. It exists
    # because the per-image worker's `max_concurrent_activities = 2` is a
    # memory ceiling, not a throughput choice, and a sub-second line fit
    # queued behind two multi-GB rawpy decodes simply expires. Name defaults
    # off `kubernetes.deployment_name`; replicas default to 1 and clamp to
    # [1, 4] in `resolve_scaling_config`.
    Validator("kubernetes.light_deployment_name", cast=str),
    Validator("kubernetes.light_active_replicas", cast=int, default=1),
    # --- GPU split + CPU fallback -------------------------------------------
    # `fishsense_data_processing_gpu_queue` is served by two Deployments
    # running the same image: one that requests a GPU and one that does not and
    # runs the same torch checkpoint on the CPU. Exactly one is scaled up at a
    # time. Both names default off `kubernetes.deployment_name`, so the trio
    # stays consistent if that is overridden. See `activities.gpu_fallback`.
    Validator("kubernetes.gpu_deployment_name", cast=str),
    Validator("kubernetes.gpu_fallback_deployment_name", cast=str),
    # CPU inference is slow per image and there is little point running several
    # in parallel; clamped to [1, 2] in `resolve_scaling_config`.
    # How many GPU pods to hold while the predict stage is active. Separate
    # from `active_replicas` (which sizes the CPU worker) because each of these
    # holds a card on a contended cluster. Clamped to [1, 4].
    Validator("kubernetes.gpu_active_replicas", cast=int, default=1),
    Validator("kubernetes.gpu_fallback_replicas", cast=int, default=1),
    # How long to wait for a pod before calling a start attempt failed. This is
    # what separates "no GPU is available" from "the pod is still pulling its
    # image" — without it every cold start would count toward the fallback.
    Validator("kubernetes.gpu_start_timeout_seconds", cast=int, default=600),
    # Failed starts before the GPU queue is handed to the CPU-only Deployment.
    # Floored at 1 in `resolve_scaling_config` so the GPU is always tried.
    Validator("kubernetes.gpu_max_start_failures", cast=int, default=3),
    # How long to stay on CPU inference before probing the GPU again. It must
    # expire: without it a single bad afternoon on NRP would strand the predict
    # stage on slow CPU inference permanently.
    Validator("kubernetes.gpu_fallback_minutes", cast=int, default=180),
    # Minimum age of a wedge before it counts as a failed start. Clamped to at
    # most `gpu_start_timeout_seconds` in `resolve_scaling_config` — a longer
    # grace would swallow every observation and the fallback could never trip.
    Validator("kubernetes.gpu_wedge_grace_minutes", cast=int, default=5),
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
    # `bucket` holds the raw/slate **scratch** (raw/, slate_pdf/). Processed
    # JPEGs that Label Studio serves live in `labels_bucket` under
    # `labels_prefix` — a separate bucket so scratch never lands in the
    # LS-facing labels bucket. `labels_bucket` defaults to `bucket` (single-
    # bucket layouts keep working); `labels_prefix` defaults to "" (no prefix).
    Validator("object_store.bucket", required=True, cast=str),
    Validator("object_store.labels_bucket", cast=str),
    Validator("object_store.labels_prefix", cast=str, default=""),
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
