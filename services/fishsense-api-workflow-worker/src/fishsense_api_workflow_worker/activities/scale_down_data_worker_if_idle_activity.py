"""Scale the NRP data-worker back to zero when its task queue is quiet.

Run hourly by ``ScaleDownIdleDataWorkerWorkflow`` (after the last
preprocess/calibration parent firing). It's the *only* thing that
scales the data-worker down — parents only ever scale up — so
overlapping parents across stages can't fight it: they all converge
on ``active_replicas``, and the sweeper drops it to 0 once nothing's
running.

"Quiet" = no workflow on ``fishsense_data_processing_queue`` is
Running *and* none has closed within ``idle_cooldown_minutes`` (so a
back-to-back dive doesn't thrash the pod up/down). The data-worker
task queue is the right signal because every data-worker workflow —
the four preprocess children, clustering, calibration, measurement,
laser-label validation — runs there; querying by task queue means
there's no workflow-type list to keep in sync.

No-op (returns ``False``) when k8s scaling isn't configured.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from fishsense_shared import build_tls_config
from temporalio import activity
from temporalio.client import Client

from fishsense_api_workflow_worker.activities.k8s_scaling import (
    apps_v1_api,
    resolve_scaling_config,
    set_deployment_replicas,
)
from fishsense_api_workflow_worker.config import settings

DATA_WORKER_TASK_QUEUE = "fishsense_data_processing_queue"


def _build_busy_query(cooldown_minutes: int) -> str:
    """Temporal list-filter matching any workflow on the data-worker task
    queue that's Running or closed within the last ``cooldown_minutes``.

    A Running workflow has no ``CloseTime``, so the ``Running`` clause
    catches in-flight ones and the ``CloseTime >`` clause catches
    recently-finished ones; an old, long-closed workflow matches
    neither.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(minutes=cooldown_minutes)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f'TaskQueue = "{DATA_WORKER_TASK_QUEUE}" '
        f'and (ExecutionStatus = "Running" or CloseTime > "{cutoff}")'
    )


async def _data_worker_task_queue_busy(cooldown_minutes: int) -> bool:
    """True iff a workflow on the data-worker task queue is Running or
    closed within the last ``cooldown_minutes``."""
    query = _build_busy_query(cooldown_minutes)
    client = await Client.connect(
        f"{settings.temporal.host}:{settings.temporal.port}",
        tls=build_tls_config(settings.temporal),
    )
    async for _ in client.list_workflows(query=query):
        return True
    return False


@activity.defn
async def scale_down_data_worker_if_idle_activity() -> bool:
    """Scale the data-worker Deployment to 0 iff its task queue is quiet.

    Returns ``True`` if it scaled down, ``False`` otherwise (still
    busy, within cooldown, or scaling disabled).
    """
    config = resolve_scaling_config()
    if config is None:
        activity.logger.info(
            "k8s scaling not configured; nothing to scale down"
        )
        return False

    if await _data_worker_task_queue_busy(config.idle_cooldown_minutes):
        activity.logger.info(
            "data-worker task queue still busy or within %d-minute cooldown; "
            "leaving %s/%s replicas as-is",
            config.idle_cooldown_minutes,
            config.namespace,
            config.deployment_name,
        )
        return False

    def _scale_to_zero() -> None:
        api = apps_v1_api(config.kubeconfig_path)
        set_deployment_replicas(api, config.namespace, config.deployment_name, 0)

    await asyncio.to_thread(_scale_to_zero)
    activity.logger.info(
        "data-worker task queue idle; scaled %s/%s to 0",
        config.namespace,
        config.deployment_name,
    )
    return True
