# pylint: disable=unnecessary-lambda,protected-access
"""Unit tests for scale_down_data_worker_if_idle_activity.

The hourly sweeper scales each NRP data-worker Deployment to 0 only when the
task queue *it* serves has had no running or recently-closed workflow. There
are three Deployments over two queues — the CPU worker, the GPU worker, and the
GPU worker's CPU-only fallback — so the per-queue pairing is itself part of
what is pinned here: a busy CPU queue must not keep a GPU pod alive.

The Temporal-busy check and the k8s client are mocked; this pins: disabled →
no-op; busy → leave replicas; wedged-but-busy → scale down anyway; idle →
scale to 0; that the sweeper never writes the GPU-fallback annotations; and
the shape of the Temporal list-filter query.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fishsense_shared import (
    DATA_PROCESSING_GPU_TASK_QUEUE,
    DATA_PROCESSING_TASK_QUEUE,
)
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    scale_down_data_worker_if_idle_activity as sut,
)
from fishsense_api_workflow_worker.activities.k8s_scaling import (
    GpuScalingConfig,
    ScalingConfig,
)

CPU_DEPLOYMENT = "fishsense-data-processing-workflow-worker"
GPU_DEPLOYMENT = f"{CPU_DEPLOYMENT}-gpu"
FALLBACK_DEPLOYMENT = f"{GPU_DEPLOYMENT}-cpu-fallback"
ALL_DEPLOYMENTS = [CPU_DEPLOYMENT, GPU_DEPLOYMENT, FALLBACK_DEPLOYMENT]


def _config(cooldown: int = 15) -> ScalingConfig:
    return ScalingConfig(
        kubeconfig_path="/tmp/nrp.kubeconfig",
        namespace="fishsense",
        deployment_name=CPU_DEPLOYMENT,
        active_replicas=1,
        idle_cooldown_minutes=cooldown,
        gpu=GpuScalingConfig(
            deployment_name=GPU_DEPLOYMENT,
            fallback_deployment_name=FALLBACK_DEPLOYMENT,
        ),
    )


def _stub_busy(monkeypatch, value: bool | set[str]) -> None:
    """Mark every queue busy/idle, or name exactly the busy queues."""
    busy = (
        {DATA_PROCESSING_TASK_QUEUE, DATA_PROCESSING_GPU_TASK_QUEUE}
        if value is True
        else (set() if value is False else value)
    )

    async def _busy(_cooldown: int, task_queue: str = DATA_PROCESSING_TASK_QUEUE):
        return task_queue in busy

    monkeypatch.setattr(sut, "_data_worker_task_queue_busy", _busy)


def _record_scales(monkeypatch) -> list:
    calls: list = []
    monkeypatch.setattr(
        sut,
        "set_deployment_replicas",
        lambda _api, ns, name, n: calls.append((ns, name, n)),
    )
    return calls


@pytest.mark.asyncio
async def test_noop_when_scaling_disabled(monkeypatch):
    monkeypatch.setattr(sut, "resolve_scaling_config", lambda: None)
    monkeypatch.setattr(
        sut,
        "_data_worker_task_queue_busy",
        lambda *_a, **_k: pytest.fail("must not query Temporal when disabled"),
    )
    result = await ActivityEnvironment().run(
        sut.scale_down_data_worker_if_idle_activity
    )
    assert result is False


@pytest.mark.asyncio
async def test_does_not_scale_down_when_busy_and_worker_is_healthy(monkeypatch):
    """Busy + a Ready pod = real work in flight. Leave it alone."""
    monkeypatch.setattr(sut, "resolve_scaling_config", lambda: _config())
    _stub_busy(monkeypatch, True)
    monkeypatch.setattr(sut, "apps_v1_api", lambda path: MagicMock())
    monkeypatch.setattr(sut, "deployment_is_wedged", lambda *_a, **_k: False)
    monkeypatch.setattr(
        sut,
        "set_deployment_replicas",
        lambda *_a, **_k: pytest.fail("must not scale a busy, healthy data-worker"),
    )

    result = await ActivityEnvironment().run(
        sut.scale_down_data_worker_if_idle_activity
    )
    assert result is False


@pytest.mark.asyncio
async def test_scales_down_a_wedged_worker_even_though_the_queue_looks_busy(
    monkeypatch,
):
    """The feedback loop this exists to break.

    A data-worker that cannot produce a Ready pod (expired Temporal cert, bad
    image, unschedulable) never drains its queue — so every dispatched child
    sits Running, the queue never looks idle, and the Deployment stays pinned at
    `active_replicas` holding GPUs around the clock. Exactly the state prod was
    in from 2026-08-14. "Busy" is only a reason to keep the pods when the pods
    can actually make progress.
    """
    monkeypatch.setattr(sut, "resolve_scaling_config", lambda: _config())
    _stub_busy(monkeypatch, True)
    monkeypatch.setattr(sut, "apps_v1_api", lambda path: MagicMock())
    monkeypatch.setattr(sut, "deployment_is_wedged", lambda *_a, **_k: True)
    calls = _record_scales(monkeypatch)
    result = await ActivityEnvironment().run(
        sut.scale_down_data_worker_if_idle_activity
    )
    assert result is True
    assert [name for _ns, name, _n in calls] == ALL_DEPLOYMENTS


@pytest.mark.asyncio
async def test_scales_every_deployment_to_zero_when_idle(monkeypatch):
    monkeypatch.setattr(sut, "resolve_scaling_config", lambda: _config())
    _stub_busy(monkeypatch, False)
    monkeypatch.setattr(sut, "apps_v1_api", lambda path: MagicMock())
    calls = _record_scales(monkeypatch)
    result = await ActivityEnvironment().run(
        sut.scale_down_data_worker_if_idle_activity
    )
    assert result is True
    assert calls == [("fishsense", name, 0) for name in ALL_DEPLOYMENTS]


@pytest.mark.asyncio
async def test_a_busy_cpu_queue_does_not_keep_the_gpu_pods_alive(monkeypatch):
    """The reason the split exists. Before it, one queue served everything, so
    hours of rectify work held a GPU the whole time."""
    monkeypatch.setattr(sut, "resolve_scaling_config", lambda: _config())
    _stub_busy(monkeypatch, {DATA_PROCESSING_TASK_QUEUE})
    monkeypatch.setattr(sut, "apps_v1_api", lambda path: MagicMock())
    monkeypatch.setattr(sut, "deployment_is_wedged", lambda *_a, **_k: False)
    calls = _record_scales(monkeypatch)
    await ActivityEnvironment().run(sut.scale_down_data_worker_if_idle_activity)
    assert [name for _ns, name, _n in calls] == [
        GPU_DEPLOYMENT,
        FALLBACK_DEPLOYMENT,
    ]


@pytest.mark.asyncio
async def test_a_busy_gpu_queue_does_not_keep_the_cpu_worker_alive(monkeypatch):
    monkeypatch.setattr(sut, "resolve_scaling_config", lambda: _config())
    _stub_busy(monkeypatch, {DATA_PROCESSING_GPU_TASK_QUEUE})
    monkeypatch.setattr(sut, "apps_v1_api", lambda path: MagicMock())
    monkeypatch.setattr(sut, "deployment_is_wedged", lambda *_a, **_k: False)
    calls = _record_scales(monkeypatch)
    await ActivityEnvironment().run(sut.scale_down_data_worker_if_idle_activity)
    assert [name for _ns, name, _n in calls] == [CPU_DEPLOYMENT]


@pytest.mark.asyncio
async def test_never_touches_the_gpu_fallback_annotations(monkeypatch):
    """Tripwire. `gpu_fallback` reads "replicas wanted, no Ready pod" as a
    failed start and counts it on the GPU Deployment's annotations. If routine
    idle scale-downs also rewrote those, a multi-hour GPU outage would reset
    its own failure counter every hour and could never reach the CPU
    fallback — silently defeating the whole feature."""
    monkeypatch.setattr(sut, "resolve_scaling_config", lambda: _config())
    _stub_busy(monkeypatch, False)
    api = MagicMock()
    monkeypatch.setattr(sut, "apps_v1_api", lambda path: api)
    await ActivityEnvironment().run(sut.scale_down_data_worker_if_idle_activity)
    api.patch_namespaced_deployment.assert_not_called()


@pytest.mark.asyncio
async def test_queries_each_task_queue_once_not_once_per_deployment(monkeypatch):
    """Two GPU Deployments share one queue; asking Temporal twice for the same
    answer is pure waste on an hourly job."""
    monkeypatch.setattr(sut, "resolve_scaling_config", lambda: _config())
    monkeypatch.setattr(sut, "apps_v1_api", lambda path: MagicMock())
    _record_scales(monkeypatch)
    asked: list[str] = []

    async def _busy(_cooldown, task_queue=DATA_PROCESSING_TASK_QUEUE):
        asked.append(task_queue)
        return False

    monkeypatch.setattr(sut, "_data_worker_task_queue_busy", _busy)
    await ActivityEnvironment().run(sut.scale_down_data_worker_if_idle_activity)
    assert sorted(asked) == sorted(
        {DATA_PROCESSING_TASK_QUEUE, DATA_PROCESSING_GPU_TASK_QUEUE}
    )


def test_busy_query_targets_data_worker_task_queue_with_running_or_recent_close():
    query = sut._build_busy_query(15)
    assert 'TaskQueue = "fishsense_data_processing_queue"' in query
    assert 'ExecutionStatus = "Running"' in query
    # A recent-close cutoff timestamp (RFC3339, Z-suffixed) is present.
    assert 'CloseTime > "20' in query and query.rstrip().endswith('Z")')


def test_busy_query_can_target_the_gpu_task_queue():
    query = sut._build_busy_query(15, DATA_PROCESSING_GPU_TASK_QUEUE)
    assert f'TaskQueue = "{DATA_PROCESSING_GPU_TASK_QUEUE}"' in query
