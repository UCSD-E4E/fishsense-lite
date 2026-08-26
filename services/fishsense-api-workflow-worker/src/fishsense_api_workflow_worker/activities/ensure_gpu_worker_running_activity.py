"""Scale up whatever can serve the data-worker's GPU queue, and say which.

The counterpart of `ensure_data_worker_running_activity` for
`fishsense_data_processing_gpu_queue`. That queue is served by two
Deployments running the same image in the same ``gpu`` role — one with a GPU,
one without — and this activity picks between them, scales it up, waits for a
pod, and returns the mode so the calling parent knows what happened:

* ``gpu`` — the GPU Deployment is up (or k8s scaling isn't configured at all,
  in which case the data-worker is assumed always-on, as it is locally).
* ``cpu_fallback`` — the GPU Deployment has failed to start too many times;
  the CPU-only Deployment is up and will run the same checkpoint slowly.
* ``unavailable`` — nothing came up inside the start timeout. **The caller must
  skip this firing**: dispatching a child onto a queue with no worker doesn't
  fail, it hangs, sitting `Running` until its execution timeout hours later.

The policy itself is pure and lives in `gpu_fallback.decide`; this module is
only the I/O around it — one Deployment read (annotations *and* readiness from
the same object, so the two can't disagree), the scale calls, the annotation
write-back, and the readiness poll.

**Why it waits.** The wait is what separates "the GPU is unavailable" from "the
pod is still pulling its image". Without it, every cold start would count
toward the fallback threshold and a healthy cluster would drift onto CPU
inference. `resolve_scaling_config` clamps the wedge grace to no more than this
timeout, so the observation that follows a timed-out wait always counts.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from temporalio import activity

from fishsense_api_workflow_worker.activities.gpu_fallback import (
    MODE_CPU_FALLBACK,
    MODE_GPU,
    MODE_UNAVAILABLE,
    GpuDecision,
    GpuState,
    decide,
)
from fishsense_api_workflow_worker.activities.k8s_scaling import (
    ScalingConfig,
    apps_v1_api,
    patch_deployment_annotations,
    readiness,
    resolve_scaling_config,
    set_deployment_replicas,
)

#: How often to re-read a Deployment while waiting for a pod to go Ready.
#: Tests set this to 0.
POLL_INTERVAL_SECONDS = 10


def _deployment_for(config: ScalingConfig, mode: str) -> str:
    return (
        config.gpu.fallback_deployment_name
        if mode == MODE_CPU_FALLBACK
        else config.gpu.deployment_name
    )


def _apply(config: ScalingConfig) -> GpuDecision:
    """One decision cycle, all of it on a worker thread.

    Reads the GPU Deployment once — the annotations carry the fallback state
    and the same object carries readiness — runs the pure policy over it, then
    writes the outcome: an absolute replica count for each Deployment (never an
    increment, so overlapping parent firings converge rather than accumulate)
    and the updated bookkeeping.
    """
    api = apps_v1_api(config.kubeconfig_path)
    deployment = api.read_namespaced_deployment(
        name=config.gpu.deployment_name, namespace=config.namespace
    )
    metadata = getattr(deployment, "metadata", None)
    state = GpuState.from_annotations(getattr(metadata, "annotations", None))
    status = readiness(deployment)

    decision = decide(
        state,
        now=datetime.now(timezone.utc),
        gpu_ready=status.ready,
        gpu_wedged=status.wedged,
        policy=config.gpu.policy,
    )

    set_deployment_replicas(
        api, config.namespace, config.gpu.deployment_name, decision.gpu_replicas
    )
    set_deployment_replicas(
        api,
        config.namespace,
        config.gpu.fallback_deployment_name,
        decision.fallback_replicas,
    )
    if decision.state != state:
        patch_deployment_annotations(
            api,
            config.namespace,
            config.gpu.deployment_name,
            decision.state.to_annotations(),
        )
    return decision


def _is_ready(config: ScalingConfig, name: str) -> bool:
    api = apps_v1_api(config.kubeconfig_path)
    return readiness(
        api.read_namespaced_deployment(name=name, namespace=config.namespace)
    ).ready


async def _wait_ready(config: ScalingConfig, name: str) -> bool:
    """Poll ``name`` until a pod is Ready or the start timeout elapses.

    Heartbeats so a long cold start doesn't look like a hung activity.
    """
    deadline = asyncio.get_running_loop().time() + config.gpu.start_timeout_seconds
    while True:
        if await asyncio.to_thread(_is_ready, config, name):
            return True
        if asyncio.get_running_loop().time() >= deadline:
            return False
        activity.heartbeat()
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


@activity.defn
async def ensure_gpu_worker_running_activity() -> str:
    """Bring up a worker for the GPU queue; return ``gpu`` / ``cpu_fallback``
    / ``unavailable``."""
    config = resolve_scaling_config()
    if config is None:
        activity.logger.info(
            "k8s scaling not configured; assuming the GPU data-worker is always-on"
        )
        return MODE_GPU

    decision = await asyncio.to_thread(_apply, config)
    activity.logger.info("GPU capacity: %s (%s)", decision.mode, decision.reason)

    if await _wait_ready(config, _deployment_for(config, decision.mode)):
        return decision.mode

    if decision.mode == MODE_GPU:
        # The wait timed out, so this start failed. Observe it again — that is
        # what increments the counter, and it may flip us to the CPU fallback.
        decision = await asyncio.to_thread(_apply, config)
        activity.logger.warning(
            "GPU data-worker did not become ready within %ds: %s",
            config.gpu.start_timeout_seconds,
            decision.reason,
        )
        if decision.mode == MODE_CPU_FALLBACK and await _wait_ready(
            config, config.gpu.fallback_deployment_name
        ):
            return MODE_CPU_FALLBACK

    # Nothing is serving the queue. Say so rather than let the caller dispatch
    # a child that would hang until its execution timeout.
    activity.logger.warning(
        "no worker could be started for the GPU queue (%s / %s); "
        "skipping this firing - the cohort selector will pick the dive up again",
        config.gpu.deployment_name,
        config.gpu.fallback_deployment_name,
    )
    return MODE_UNAVAILABLE
