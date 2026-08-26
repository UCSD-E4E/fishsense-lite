"""Scale the NRP data-worker Deployments back to zero when their queues are quiet.

Run hourly by ``ScaleDownIdleDataWorkerWorkflow`` (after the last
preprocess/calibration parent firing). It's the *only* thing that
scales the data-worker down — parents only ever scale up — so
overlapping parents across stages can't fight it: they all converge
on ``active_replicas``, and the sweeper drops it to 0 once nothing's
running.

There are three Deployments across two task queues (see
`k8s_scaling.ScalingConfig.sweep_targets`): the CPU worker on
`fishsense_data_processing_queue`, and the GPU worker plus its CPU-only
fallback, which both serve `fishsense_data_processing_gpu_queue`. Each
Deployment is swept against the queue *it* serves, so a busy CPU queue never
keeps a GPU pod alive and vice versa — the point of having split them.

"Quiet" = no workflow on that task queue is Running *and* none has
closed within ``idle_cooldown_minutes`` (so a back-to-back dive doesn't
thrash the pod up/down). Querying by task queue means there's no
workflow-type list to keep in sync.

The sweeper writes **only** the scale subresource, never the GPU-fallback
annotations. That separation is load-bearing: `gpu_fallback` reads
`spec.replicas > 0 && no ready pod` as a failed start, and if scaling down for
ordinary idleness also cleared the bookkeeping, a long GPU outage would keep
resetting its own failure counter every hour and could never reach the
fallback. `tests/test_scale_down_data_worker_if_idle_activity.py` pins it.

No-op (returns ``False``) when k8s scaling isn't configured.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from fishsense_shared import (
    DATA_PROCESSING_TASK_QUEUE,
    build_tls_config,
    temporal_namespace,
)
from temporalio import activity
from temporalio.client import Client

from fishsense_api_workflow_worker.activities.k8s_scaling import (
    apps_v1_api,
    deployment_is_wedged,
    resolve_scaling_config,
    set_deployment_replicas,
)
from fishsense_api_workflow_worker.config import settings

# Retained under its original name for the integration test and any operator
# script that imports it; the CPU worker's queue.
DATA_WORKER_TASK_QUEUE = DATA_PROCESSING_TASK_QUEUE


def _build_busy_query(
    cooldown_minutes: int, task_queue: str = DATA_WORKER_TASK_QUEUE
) -> str:
    """Temporal list-filter matching any workflow on ``task_queue`` that's
    Running or closed within the last ``cooldown_minutes``.

    A Running workflow has no ``CloseTime``, so the ``Running`` clause
    catches in-flight ones and the ``CloseTime >`` clause catches
    recently-finished ones; an old, long-closed workflow matches
    neither.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(minutes=cooldown_minutes)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f'TaskQueue = "{task_queue}" '
        f'and (ExecutionStatus = "Running" or CloseTime > "{cutoff}")'
    )


async def _data_worker_task_queue_busy(
    cooldown_minutes: int, task_queue: str = DATA_WORKER_TASK_QUEUE
) -> bool:
    """True iff a workflow on ``task_queue`` is Running or closed within the
    last ``cooldown_minutes``."""
    query = _build_busy_query(cooldown_minutes, task_queue)
    client = await Client.connect(
        f"{settings.temporal.host}:{settings.temporal.port}",
        tls=build_tls_config(settings.temporal),
        namespace=temporal_namespace(settings.temporal),
    )
    async for _ in client.list_workflows(query=query):
        return True
    return False


async def _busy_task_queues(
    task_queues: tuple[str, ...], cooldown_minutes: int
) -> set[str]:
    """Which of ``task_queues`` are busy. Each is queried once, however many
    Deployments serve it."""
    return {
        task_queue
        for task_queue in task_queues
        if await _data_worker_task_queue_busy(cooldown_minutes, task_queue)
    }


@activity.defn
async def scale_down_data_worker_if_idle_activity() -> bool:
    """Scale each data-worker Deployment to 0 iff the queue it serves is quiet.

    Returns ``True`` if it scaled anything down, ``False`` otherwise (all
    still busy, within cooldown, or scaling disabled).
    """
    config = resolve_scaling_config()
    if config is None:
        activity.logger.info("k8s scaling not configured; nothing to scale down")
        return False

    targets = config.sweep_targets()
    busy = await _busy_task_queues(
        tuple({task_queue for _, task_queue in targets}),
        config.idle_cooldown_minutes,
    )

    def _sweep() -> list[tuple[str, str]]:
        """Return the (deployment, outcome) pairs. One k8s client for the whole
        pass — the wedge checks and the scale calls share it."""
        api = apps_v1_api(config.kubeconfig_path)
        outcomes = []
        for name, task_queue in targets:
            if task_queue in busy and not deployment_is_wedged(
                api, config.namespace, name
            ):
                outcomes.append((name, "busy"))
                continue
            outcomes.append((name, "wedged" if task_queue in busy else "idle"))
            set_deployment_replicas(api, config.namespace, name, 0)
        return outcomes

    outcomes = await asyncio.to_thread(_sweep)

    for name, outcome in outcomes:
        if outcome == "busy":
            activity.logger.info(
                "task queue for %s/%s still busy or within %d-minute cooldown "
                "and the worker is healthy; leaving replicas as-is",
                config.namespace,
                name,
                config.idle_cooldown_minutes,
            )
        elif outcome == "wedged":
            # WARNING, not info: reclaiming the pods bounds the cost but fixes
            # nothing. Something is stopping this worker from starting, and the
            # queue backlog this leaves behind is real work that is not
            # happening. For the GPU Deployment this is also what
            # `ensure_gpu_worker_running_activity` counts toward the CPU
            # fallback, so a persistent wedge there does eventually route
            # around itself.
            activity.logger.warning(
                "data-worker %s/%s has no Ready pod but its task queue is busy - "
                "it cannot drain and was holding replicas for nothing; scaled to 0. "
                "Check pod status (CrashLoopBackOff? expired Temporal cert? "
                "unschedulable?) - the next parent with work will scale it back up.",
                config.namespace,
                name,
            )
        else:
            activity.logger.info(
                "task queue idle; scaled %s/%s to 0", config.namespace, name
            )

    return any(outcome != "busy" for _, outcome in outcomes)
