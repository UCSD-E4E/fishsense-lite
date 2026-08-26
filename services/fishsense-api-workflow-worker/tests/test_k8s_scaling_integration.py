# pylint: disable=unused-argument
"""K8s integration tests for the data-worker scaling activities.

Marked ``@pytest.mark.k8s`` — needs a Kubernetes cluster reachable via
``$KUBECONFIG`` with the data-worker Deployment already applied. CI's
`k8s-tests.yml` does `kind create cluster` + creates stub Secrets +
`kubectl apply -k deploy/k8s/data-worker` (which is itself a server-side
validation of the manifests) before running `pytest -m k8s`. Skipped
when ``$KUBECONFIG`` isn't set, so `pytest -m k8s` is a harmless no-op
locally without a cluster.

Covers what the unit tests mock away: `ensure_data_worker_running_activity`
and `scale_down_data_worker_if_idle_activity` patching a real
Deployment's ``.spec.replicas``, `apps_v1_api`/`load_kube_config`
working against a real kubeconfig, `resolve_scaling_config` against
real-ish settings, and the GPU fallback's annotation round-trip — where a
``None`` value has to actually REMOVE the key, which only a real apiserver
can confirm. The Temporal `list_workflows` query that `scale_down` uses is
mocked here and exercised against a real Temporal in
`test_scale_down_query_integration.py`.

**Which Deployment a wedge test points at is load-bearing.** kind has no GPU
nodes, so `...-worker-gpu` — which requests `nvidia.com/gpu: 1` and carries a
compute-capability node affinity — is *permanently* Unschedulable there. That
is a deterministic, honest wedge. The CPU Deployment used to be wedged for the
same reason, but as of the GPU/CPU split it requests no GPU: it is now merely
slow to fail (big image pull, then a Temporal connect against stub certs), so
asserting it is wedged would be a race. The wedge tests therefore use the GPU
Deployment.
"""

from __future__ import annotations

import os

import pytest
from kubernetes.client.rest import ApiException
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    ensure_data_worker_running_activity as ensure_mod,
    ensure_gpu_worker_running_activity as ensure_gpu_mod,
    gpu_fallback,
    k8s_scaling,
    scale_down_data_worker_if_idle_activity as scale_down_mod,
)

pytestmark = pytest.mark.k8s

DEPLOYMENT = "fishsense-data-processing-workflow-worker"
# Permanently Unschedulable on kind (GPU request + compute-capability
# affinity), which is exactly what makes it a deterministic wedge fixture.
GPU_DEPLOYMENT = f"{DEPLOYMENT}-gpu"
GPU_FALLBACK_DEPLOYMENT = f"{GPU_DEPLOYMENT}-cpu-fallback"


@pytest.fixture
def kubeconfig() -> str:
    path = os.environ.get("KUBECONFIG")
    if not path or not os.path.isfile(path):
        pytest.skip("KUBECONFIG not set / not a file — run this via the k8s CI job")
    return path


@pytest.fixture
def namespace() -> str:
    # `kind create cluster` puts everything in `default`.
    return os.environ.get("K8S_NAMESPACE", "default")


@pytest.fixture
def configure_scaling(monkeypatch, kubeconfig, namespace):
    """Point the scaling helpers at the test cluster + Deployment.

    Skips (rather than fails) when the Deployment isn't present — that
    means `kubectl apply -k deploy/k8s/data-worker` hasn't run against
    this cluster yet. CI's k8s-tests.yml does that before `pytest -m
    k8s`, so it only skips when someone runs this against a bare
    cluster.
    """
    monkeypatch.setattr(
        k8s_scaling,
        "settings",
        {
            "kubernetes": {
                "kubeconfig_path": kubeconfig,
                "namespace": namespace,
                "deployment_name": DEPLOYMENT,
                "active_replicas": 2,
                "idle_cooldown_minutes": 0,
                # No waiting for a pod: on kind nothing ever becomes Ready, so
                # the default 600s would just stall the job. With the grace at
                # 0 and a single permitted failure, the fallback flip is
                # reached in a bounded number of calls.
                "gpu_start_timeout_seconds": 0,
                "gpu_wedge_grace_minutes": 0,
                "gpu_max_start_failures": 1,
            }
        },
    )
    try:
        k8s_scaling.apps_v1_api(kubeconfig).read_namespaced_deployment(
            DEPLOYMENT, namespace
        )
    except ApiException as exc:
        if exc.status == 404:
            pytest.skip(
                f"Deployment {namespace}/{DEPLOYMENT} not found — "
                "run `kubectl apply -k deploy/k8s/data-worker` first"
            )
        raise


def _api(kubeconfig: str):
    return k8s_scaling.apps_v1_api(kubeconfig)


def _replicas(kubeconfig: str, namespace: str, name: str = DEPLOYMENT) -> int:
    # Read the Deployment's .spec.replicas (a `*int32`, so 0 round-trips),
    # NOT the Scale subresource — autoscaling/v1 ScaleSpec.replicas is a
    # plain int32 with `omitempty`, so the API omits it when the count is
    # 0 and the Python client deserializes that as None.
    return _api(kubeconfig).read_namespaced_deployment(name, namespace).spec.replicas


def _set_replicas(
    kubeconfig: str, namespace: str, n: int, name: str = DEPLOYMENT
) -> None:
    k8s_scaling.set_deployment_replicas(_api(kubeconfig), namespace, name, n)


def _annotations(kubeconfig: str, namespace: str, name: str) -> dict:
    meta = _api(kubeconfig).read_namespaced_deployment(name, namespace).metadata
    return meta.annotations or {}


async def test_ensure_running_scales_the_real_deployment_up(
    configure_scaling, kubeconfig, namespace
):
    _set_replicas(kubeconfig, namespace, 0)
    result = await ActivityEnvironment().run(
        ensure_mod.ensure_data_worker_running_activity
    )
    assert result == 2
    assert _replicas(kubeconfig, namespace) == 2


async def test_scale_down_when_idle_zeroes_the_real_deployment(
    configure_scaling, kubeconfig, namespace, monkeypatch
):
    async def _not_busy(_cooldown: int, _task_queue: str = "") -> bool:
        return False

    monkeypatch.setattr(scale_down_mod, "_data_worker_task_queue_busy", _not_busy)
    _set_replicas(kubeconfig, namespace, 3)
    result = await ActivityEnvironment().run(
        scale_down_mod.scale_down_data_worker_if_idle_activity
    )
    assert result is True
    assert _replicas(kubeconfig, namespace) == 0


async def test_scale_down_leaves_a_busy_healthy_deployment_alone(
    configure_scaling, kubeconfig, namespace, monkeypatch
):
    """Busy + able to make progress = leave it running.

    `deployment_is_wedged` is stubbed False here because this cluster cannot
    produce the honest version of "healthy": nothing ever reaches Ready on
    kind — the GPU Deployment is permanently Unschedulable, and the other two
    would have to pull a multi-GB image and then reach krg-prod Temporal with
    stub certs. The wedge test below needs no such stub, which is what makes
    it real rather than simulated.
    """
    async def _busy(_cooldown: int, _task_queue: str = "") -> bool:
        return True

    monkeypatch.setattr(scale_down_mod, "_data_worker_task_queue_busy", _busy)
    monkeypatch.setattr(scale_down_mod, "deployment_is_wedged", lambda *_a, **_k: False)
    _set_replicas(kubeconfig, namespace, 1)
    result = await ActivityEnvironment().run(
        scale_down_mod.scale_down_data_worker_if_idle_activity
    )
    assert result is False
    assert _replicas(kubeconfig, namespace) == 1


async def test_deployment_is_wedged_against_a_real_apiserver(
    configure_scaling, kubeconfig, namespace
):
    """Pins the field semantics the wedge check depends on.

    The predicate hinges on `status.readyReplicas` being ABSENT (deserialized
    as None) rather than 0 when nothing is Ready — a real apiserver is the only
    thing that can confirm that, and reading None as "unknown, assume healthy"
    is precisely what pinned prod's GPUs from 2026-08-14.

    Uses the GPU Deployment: its pods are Unschedulable on kind for a
    structural reason (no node satisfies `nvidia.com/gpu`), so "no Ready pod"
    holds immediately and forever rather than only until an image finishes
    pulling.
    """
    _set_replicas(kubeconfig, namespace, 1, GPU_DEPLOYMENT)
    assert k8s_scaling.deployment_is_wedged(
        _api(kubeconfig), namespace, GPU_DEPLOYMENT
    )

    # Scaled to zero, nothing is held, so there is nothing to reclaim.
    _set_replicas(kubeconfig, namespace, 0, GPU_DEPLOYMENT)
    assert not k8s_scaling.deployment_is_wedged(
        _api(kubeconfig), namespace, GPU_DEPLOYMENT
    )


async def test_scale_down_reclaims_a_wedged_busy_deployment(
    configure_scaling, kubeconfig, namespace, monkeypatch
):
    """End-to-end reproduction of the prod feedback loop, unmocked.

    The queue reports busy AND the Deployment cannot produce a Ready pod — in
    kind because the GPU request is unschedulable, in prod from 2026-08-14
    because the Temporal client cert had expired and every pod crash-looped.
    Before this behaviour existed the sweeper read "busy" and left the replicas
    up, so the Deployment held NRP GPUs indefinitely while draining nothing.

    Asserted on the GPU Deployment because that is the one kind wedges
    deterministically; the sweep covers all three in the same pass.
    """
    async def _busy(_cooldown: int, _task_queue: str = "") -> bool:
        return True

    monkeypatch.setattr(scale_down_mod, "_data_worker_task_queue_busy", _busy)
    _set_replicas(kubeconfig, namespace, 2, GPU_DEPLOYMENT)
    result = await ActivityEnvironment().run(
        scale_down_mod.scale_down_data_worker_if_idle_activity
    )
    assert result is True
    assert _replicas(kubeconfig, namespace, GPU_DEPLOYMENT) == 0


async def test_annotation_patch_removes_a_key_on_a_real_apiserver(
    configure_scaling, kubeconfig, namespace
):
    """`GpuState.to_annotations()` returns None for every field it wants gone,
    and the fallback's whole "absent means healthy" contract rests on a None
    value actually REMOVING the key rather than storing an empty string.

    That is strategic-merge-patch semantics, so a mock cannot confirm it — the
    unit tests assert the call shape, and this asserts the apiserver agrees.
    """
    api = _api(kubeconfig)
    k8s_scaling.patch_deployment_annotations(
        api, namespace, GPU_DEPLOYMENT, {"fishsense.e4e.ucsd.edu/itest": "1"}
    )
    assert _annotations(kubeconfig, namespace, GPU_DEPLOYMENT).get(
        "fishsense.e4e.ucsd.edu/itest"
    ) == "1"

    k8s_scaling.patch_deployment_annotations(
        api, namespace, GPU_DEPLOYMENT, {"fishsense.e4e.ucsd.edu/itest": None}
    )
    assert (
        "fishsense.e4e.ucsd.edu/itest"
        not in _annotations(kubeconfig, namespace, GPU_DEPLOYMENT)
    )


async def test_gpu_worker_falls_back_to_cpu_against_a_real_cluster(
    configure_scaling, kubeconfig, namespace
):
    """The headline behaviour, end to end on a real apiserver.

    kind can never schedule the GPU Deployment, which is precisely the prod
    condition the fallback exists for (no free Turing-or-newer card, exhausted
    quota, drained pool). Repeated calls must therefore give up on it and bring
    the CPU-only Deployment up instead.

    Asserts replica counts rather than the returned mode: nothing reaches Ready
    on kind, so the activity honestly reports `unavailable` throughout. What
    matters — and what only a real apiserver proves — is that the flip was
    persisted, driven by annotations it read back from the cluster.
    """
    _set_replicas(kubeconfig, namespace, 0, GPU_DEPLOYMENT)
    _set_replicas(kubeconfig, namespace, 0, GPU_FALLBACK_DEPLOYMENT)
    # Start from a clean slate — an empty state clears all three annotations.
    k8s_scaling.patch_deployment_annotations(
        _api(kubeconfig),
        namespace,
        GPU_DEPLOYMENT,
        gpu_fallback.GpuState().to_annotations(),
    )

    for _ in range(5):
        await ActivityEnvironment().run(
            ensure_gpu_mod.ensure_gpu_worker_running_activity
        )
        if _replicas(kubeconfig, namespace, GPU_FALLBACK_DEPLOYMENT) == 1:
            break
    else:
        pytest.fail("the GPU queue never fell back to the CPU Deployment")

    assert _replicas(kubeconfig, namespace, GPU_DEPLOYMENT) == 0
    assert (
        gpu_fallback.FALLBACK_UNTIL_ANNOTATION
        in _annotations(kubeconfig, namespace, GPU_DEPLOYMENT)
    )
