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
working against a real kubeconfig, and `resolve_scaling_config` against
real-ish settings. The Temporal `list_workflows` query that
`scale_down` uses is mocked here and exercised against a real Temporal
in `test_scale_down_query_integration.py`.
"""

from __future__ import annotations

import os

import pytest
from kubernetes.client.rest import ApiException
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    ensure_data_worker_running_activity as ensure_mod,
    k8s_scaling,
    scale_down_data_worker_if_idle_activity as scale_down_mod,
)

pytestmark = pytest.mark.k8s

DEPLOYMENT = "fishsense-data-processing-workflow-worker"


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


def _replicas(kubeconfig: str, namespace: str) -> int:
    # Read the Deployment's .spec.replicas (a `*int32`, so 0 round-trips),
    # NOT the Scale subresource — autoscaling/v1 ScaleSpec.replicas is a
    # plain int32 with `omitempty`, so the API omits it when the count is
    # 0 and the Python client deserializes that as None.
    return _api(kubeconfig).read_namespaced_deployment(DEPLOYMENT, namespace).spec.replicas


def _set_replicas(kubeconfig: str, namespace: str, n: int) -> None:
    k8s_scaling.set_deployment_replicas(_api(kubeconfig), namespace, DEPLOYMENT, n)


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
    async def _not_busy(_cooldown: int) -> bool:
        return False

    monkeypatch.setattr(scale_down_mod, "_data_worker_task_queue_busy", _not_busy)
    _set_replicas(kubeconfig, namespace, 3)
    result = await ActivityEnvironment().run(
        scale_down_mod.scale_down_data_worker_if_idle_activity
    )
    assert result is True
    assert _replicas(kubeconfig, namespace) == 0


async def test_scale_down_leaves_a_busy_deployment_alone(
    configure_scaling, kubeconfig, namespace, monkeypatch
):
    async def _busy(_cooldown: int) -> bool:
        return True

    monkeypatch.setattr(scale_down_mod, "_data_worker_task_queue_busy", _busy)
    _set_replicas(kubeconfig, namespace, 1)
    result = await ActivityEnvironment().run(
        scale_down_mod.scale_down_data_worker_if_idle_activity
    )
    assert result is False
    assert _replicas(kubeconfig, namespace) == 1
