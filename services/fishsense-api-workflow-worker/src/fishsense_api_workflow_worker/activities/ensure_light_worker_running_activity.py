"""Scale the NRP light-queue worker up before dispatching a child workflow.

The counterpart of `ensure_data_worker_running_activity` for
``fishsense_data_processing_light_queue`` — the queue the stages that hold no
image bytes run on: frame clustering, laser calibration, fish measurement,
laser depth, laser-label validation and the auto-accept gate.

**Why they are not simply on the per-image queue.** They were, until
2026-09-04. That queue's worker runs ``general.max_concurrent_activities = 2``,
and the 2 is a memory ceiling rather than a throughput choice: each per-image
activity decodes a full-res `.ORF` and peaks at 1-3 GB, the Temporal default of
100 OOMKilled the pod into CrashLoopBackOff, and a cap of 4 did it again with
17 restarts. So two slots is the most that pod can safely offer, one 34-image
preprocess dispatch takes both for as long as it takes, and everything else
waits. That expired three of the auto-accept drain's first four firings and two
consecutive laser calibrations, all with `ScheduleToStart timeout`, for work
that takes under a second.

Splitting the queue is what decouples the two: this worker holds no image
bytes, so its pod is small, its concurrency cap is unrelated to the decoders',
and nothing a preprocess dive does can starve it.

Same contract as the per-image wake: called only once the parent knows there is
real work, scales to an absolute target (idempotent — overlapping parents
converge on the count rather than each adding a pod), and returns immediately
so the pod's cold start overlaps the parent's remaining steps. That cold start
is the cost of NRP's scale-to-zero requirement, and it is why the light stages'
activities bound queue wait separately from execution — see
`fishsense_shared.auto_accept_timeouts` for the worked example.

No-op (returns 0) when k8s scaling isn't configured — the data-worker is then
assumed always-on, which is the pre-NRP behavior and what the local
devcontainer does with ``role = "all"``.
"""

from __future__ import annotations

import asyncio

from temporalio import activity

from fishsense_api_workflow_worker.activities.k8s_scaling import (
    apps_v1_api,
    resolve_scaling_config,
    set_deployment_replicas,
)


@activity.defn
async def ensure_light_worker_running_activity() -> int:
    """Scale the light-queue Deployment up to ``light_active_replicas``.

    Returns the replica count it targeted, or 0 when scaling is disabled (no
    kubeconfig configured).
    """
    config = resolve_scaling_config()
    if config is None:
        activity.logger.info(
            "k8s scaling not configured; assuming light worker is always-on"
        )
        return 0

    def _scale() -> None:
        api = apps_v1_api(config.kubeconfig_path)
        set_deployment_replicas(
            api,
            config.namespace,
            config.light.deployment_name,
            config.light.active_replicas,
        )

    await asyncio.to_thread(_scale)
    activity.logger.info(
        "scaled light worker %s/%s to %d replica(s)",
        config.namespace,
        config.light.deployment_name,
        config.light.active_replicas,
    )
    return config.light.active_replicas
