"""Scale the NRP data-worker up before dispatching a child workflow.

Called by every parent workflow that dispatches to
``fishsense_data_processing_queue`` — but only once the parent knows
there's real work (a non-empty image set / a selected dive), so a
quiet hour never wakes the pod. Scales the data-worker Deployment to
the configured ``active_replicas`` (an absolute target, idempotent —
overlapping parents across stages converge on the same count rather
than each adding a pod) and returns immediately; the parent's
``execute_child_workflow`` naturally waits out the pod's cold start.

No-op (returns 0) when k8s scaling isn't configured — the data-worker
is then assumed always-on, which is the pre-NRP behavior and what the
local devcontainer does.
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
async def ensure_data_worker_running_activity() -> int:
    """Scale the data-worker Deployment up to ``active_replicas``.

    Returns the replica count it targeted, or 0 when scaling is
    disabled (no kubeconfig configured).
    """
    config = resolve_scaling_config()
    if config is None:
        activity.logger.info(
            "k8s scaling not configured; assuming data-worker is always-on"
        )
        return 0

    def _scale() -> None:
        api = apps_v1_api(config.kubeconfig_path)
        set_deployment_replicas(
            api, config.namespace, config.deployment_name, config.active_replicas
        )

    await asyncio.to_thread(_scale)
    activity.logger.info(
        "scaled data-worker %s/%s to %d replica(s)",
        config.namespace,
        config.deployment_name,
        config.active_replicas,
    )
    return config.active_replicas
