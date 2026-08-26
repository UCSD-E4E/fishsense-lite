"""Unit tests for ensure_gpu_worker_running_activity.

Kubernetes is mocked. What is pinned here is the contract the predict parents
depend on: which Deployment ends up scaled up, what mode the parent is told to
expect, and — the point of the whole feature — that a GPU that will not start
eventually hands its queue to the CPU-only Deployment instead of stalling.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    ensure_gpu_worker_running_activity as sut,
)
from fishsense_api_workflow_worker.activities.gpu_fallback import (
    FAILURES_ANNOTATION,
    FALLBACK_UNTIL_ANNOTATION,
    MODE_CPU_FALLBACK,
    MODE_GPU,
    MODE_UNAVAILABLE,
    FallbackPolicy,
)
from fishsense_api_workflow_worker.activities.k8s_scaling import (
    GpuScalingConfig,
    ScalingConfig,
)

GPU = "fishsense-data-processing-workflow-worker-gpu"
FALLBACK = "fishsense-data-processing-workflow-worker-gpu-cpu-fallback"


def _config(**gpu_overrides) -> ScalingConfig:
    gpu = {
        "deployment_name": GPU,
        "fallback_deployment_name": FALLBACK,
        # 0 = one readiness check with no waiting. Cheap, and it means a test
        # says what is broken via `_Cluster(broken=...)` rather than by timing.
        "start_timeout_seconds": 0,
        "policy": FallbackPolicy(
            active_replicas=1,
            fallback_replicas=1,
            max_start_failures=3,
            wedge_grace=timedelta(0),
            fallback_window=timedelta(hours=3),
        ),
    }
    gpu.update(gpu_overrides)
    return ScalingConfig(
        kubeconfig_path="/tmp/nrp.kubeconfig",
        namespace="fishsense",
        deployment_name="fishsense-data-processing-workflow-worker",
        active_replicas=1,
        idle_cooldown_minutes=15,
        gpu=GpuScalingConfig(**gpu),
    )


class _Cluster:
    """A tiny stand-in for the two Deployments this activity drives.

    Pods go Ready as soon as they are asked for, which models a healthy
    cluster — so each test only has to say what is *broken*, via ``broken``.
    A Deployment listed there accepts a replica count and never reports a Ready
    pod, which is exactly the shape of an unschedulable GPU request, an
    exhausted quota, or a CrashLoopBackOff.
    """

    def __init__(self, broken: set[str] | None = None):
        self.broken = broken or set()
        self.replicas = {GPU: 0, FALLBACK: 0}
        self.ready = {GPU: 0, FALLBACK: 0}
        self.annotations: dict[str, dict[str, str]] = {GPU: {}, FALLBACK: {}}
        self.patched_bodies: list[dict] = []

    def _set_replicas(self, name: str, count: int) -> None:
        self.replicas[name] = count
        self.ready[name] = 0 if name in self.broken else count

    def api(self):
        api = MagicMock()

        def _read(name, namespace):  # pylint: disable=unused-argument
            return SimpleNamespace(
                metadata=SimpleNamespace(annotations=dict(self.annotations[name])),
                spec=SimpleNamespace(replicas=self.replicas[name]),
                status=SimpleNamespace(ready_replicas=self.ready[name] or None),
            )

        def _scale(name, namespace, body):  # pylint: disable=unused-argument
            self._set_replicas(name, body["spec"]["replicas"])

        def _patch(name, namespace, body):  # pylint: disable=unused-argument
            self.patched_bodies.append(body)
            for key, value in body["metadata"]["annotations"].items():
                if value is None:
                    self.annotations[name].pop(key, None)
                else:
                    self.annotations[name][key] = value

        api.read_namespaced_deployment.side_effect = _read
        api.patch_namespaced_deployment_scale.side_effect = _scale
        api.patch_namespaced_deployment.side_effect = _patch
        return api


def _install(monkeypatch, cluster, config):
    monkeypatch.setattr(sut, "resolve_scaling_config", lambda: config)
    monkeypatch.setattr(sut, "apps_v1_api", lambda _path: cluster.api())


async def _run():
    return await ActivityEnvironment().run(sut.ensure_gpu_worker_running_activity)


@pytest.mark.asyncio
async def test_noop_when_scaling_is_not_configured(monkeypatch):
    """Local devcontainer / pre-NRP: the worker is assumed always-on, so the
    parent should dispatch to the GPU queue exactly as before."""
    monkeypatch.setattr(sut, "resolve_scaling_config", lambda: None)
    monkeypatch.setattr(
        sut, "apps_v1_api", lambda *_a: pytest.fail("must not touch k8s")
    )
    assert await _run() == MODE_GPU


@pytest.mark.asyncio
async def test_cold_start_scales_the_gpu_deployment_up(monkeypatch):
    cluster = _Cluster()
    _install(monkeypatch, cluster, _config())
    assert await _run() == MODE_GPU
    assert cluster.replicas[GPU] == 1
    assert cluster.replicas[FALLBACK] == 0


@pytest.mark.asyncio
async def test_a_ready_gpu_keeps_the_fallback_down(monkeypatch):
    cluster = _Cluster()
    cluster.replicas[GPU], cluster.ready[GPU] = 1, 1
    _install(monkeypatch, cluster, _config())
    assert await _run() == MODE_GPU
    assert cluster.replicas[FALLBACK] == 0


@pytest.mark.asyncio
async def test_repeated_failed_starts_hand_the_queue_to_the_cpu_fallback(
    monkeypatch,
):
    """The headline behavior: a GPU that never schedules must not stall the
    predict stage forever.

    Bounded so the test fails loudly rather than looping if the fallback ever
    stops tripping. The bound is generous on purpose — what matters is that it
    converges within a couple of hourly firings, not the exact count.
    """
    cluster = _Cluster(broken={GPU})
    config = _config()
    _install(monkeypatch, cluster, config)

    firings = 0
    while True:
        firings += 1
        mode = await _run()
        if mode == MODE_CPU_FALLBACK:
            break
        assert firings <= config.gpu.policy.max_start_failures + 1, (
            "the GPU queue never fell back to CPU"
        )

    assert cluster.replicas[GPU] == 0
    assert cluster.replicas[FALLBACK] == 1
    assert FALLBACK_UNTIL_ANNOTATION in cluster.annotations[GPU]


@pytest.mark.asyncio
async def test_the_fallback_window_is_held_then_released(monkeypatch):
    cluster = _Cluster()
    config = _config()
    _install(monkeypatch, cluster, config)

    # Already in fallback, window still open.
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    cluster.annotations[GPU][FALLBACK_UNTIL_ANNOTATION] = future.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    assert await _run() == MODE_CPU_FALLBACK
    assert cluster.replicas[FALLBACK] == 1
    assert cluster.replicas[GPU] == 0

    # Window expired → probe the GPU again, and drop the fallback.
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    cluster.annotations[GPU][FALLBACK_UNTIL_ANNOTATION] = past.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    assert await _run() == MODE_GPU
    assert cluster.replicas[GPU] == 1
    assert cluster.replicas[FALLBACK] == 0
    assert FALLBACK_UNTIL_ANNOTATION not in cluster.annotations[GPU]
    assert FAILURES_ANNOTATION not in cluster.annotations[GPU]


@pytest.mark.asyncio
async def test_reports_unavailable_when_the_fallback_cannot_start_either(
    monkeypatch,
):
    """Nothing can serve the queue. The parent must be told, so it skips this
    firing instead of dispatching a child that would hang until its execution
    timeout hours later."""
    cluster = _Cluster(broken={GPU, FALLBACK})
    config = _config()
    _install(monkeypatch, cluster, config)
    cluster.annotations[GPU][FALLBACK_UNTIL_ANNOTATION] = (
        datetime.now(timezone.utc) + timedelta(hours=1)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert await _run() == MODE_UNAVAILABLE


@pytest.mark.asyncio
async def test_waits_for_a_pod_before_declaring_the_start_failed(monkeypatch):
    """A cold pod that becomes Ready inside the start timeout is a success, not
    a failed attempt — otherwise every image pull would count toward the
    fallback threshold."""
    cluster = _Cluster(broken={GPU})
    _install(monkeypatch, cluster, _config(start_timeout_seconds=60))
    monkeypatch.setattr(sut, "POLL_INTERVAL_SECONDS", 0)

    api = cluster.api()
    inner_read = api.read_namespaced_deployment.side_effect
    polls = {"n": 0}

    def _read(name, namespace):
        """The pod finishes pulling its image on the third poll."""
        polls["n"] += 1
        if polls["n"] >= 3:
            cluster.broken.discard(GPU)
            cluster.ready[GPU] = cluster.replicas[GPU]
        return inner_read(name=name, namespace=namespace)

    api.read_namespaced_deployment.side_effect = _read
    monkeypatch.setattr(sut, "apps_v1_api", lambda _path: api)

    assert await _run() == MODE_GPU
    assert FAILURES_ANNOTATION not in cluster.annotations[GPU]


@pytest.mark.asyncio
async def test_never_patches_a_pod_template(monkeypatch):
    """Tripwire. Annotation writes go on the Deployment's own metadata; putting
    them on `spec.template` would roll the pods on every firing — restarting
    the very worker we are trying to get started."""
    cluster = _Cluster(broken={GPU})
    _install(monkeypatch, cluster, _config())
    cluster.replicas[GPU] = 1
    await _run()
    assert cluster.patched_bodies, "expected at least one annotation patch"
    for body in cluster.patched_bodies:
        assert set(body) == {"metadata"}
        assert set(body["metadata"]) == {"annotations"}
