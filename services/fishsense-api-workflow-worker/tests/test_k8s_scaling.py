"""Unit tests for the shared NRP data-worker scaling helpers.

These don't touch Kubernetes or Temporal — they pin the config
resolution (disabled-by-default, required namespace, clamped replica
count, defaults) and the declarative replica-set call shape.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fishsense_api_workflow_worker.activities import k8s_scaling


def test_disabled_when_no_kubeconfig(monkeypatch):
    monkeypatch.setattr(k8s_scaling, "settings", {})
    assert k8s_scaling.resolve_scaling_config() is None

    # Section present but no kubeconfig_path → still disabled.
    monkeypatch.setattr(k8s_scaling, "settings", {"kubernetes": {"namespace": "ns"}})
    assert k8s_scaling.resolve_scaling_config() is None


def test_requires_namespace_when_kubeconfig_set(monkeypatch):
    monkeypatch.setattr(
        k8s_scaling, "settings", {"kubernetes": {"kubeconfig_path": "/tmp/kc"}}
    )
    with pytest.raises(ValueError):
        k8s_scaling.resolve_scaling_config()


def test_defaults_when_only_required_keys_set(monkeypatch):
    monkeypatch.setattr(
        k8s_scaling,
        "settings",
        {"kubernetes": {"kubeconfig_path": "/tmp/kc", "namespace": "ns"}},
    )
    cfg = k8s_scaling.resolve_scaling_config()
    assert cfg is not None
    assert cfg.kubeconfig_path == "/tmp/kc"
    assert cfg.namespace == "ns"
    assert cfg.deployment_name == k8s_scaling.DEFAULT_DEPLOYMENT_NAME
    assert cfg.active_replicas == 1
    assert cfg.idle_cooldown_minutes == 15


def test_active_replicas_clamped_to_ceiling(monkeypatch):
    monkeypatch.setattr(
        k8s_scaling,
        "settings",
        {
            "kubernetes": {
                "kubeconfig_path": "/tmp/kc",
                "namespace": "ns",
                "active_replicas": 99,
            }
        },
    )
    assert (
        k8s_scaling.resolve_scaling_config().active_replicas
        == k8s_scaling.MAX_ACTIVE_REPLICAS
    )


def test_active_replicas_clamped_to_floor(monkeypatch):
    monkeypatch.setattr(
        k8s_scaling,
        "settings",
        {
            "kubernetes": {
                "kubeconfig_path": "/tmp/kc",
                "namespace": "ns",
                "active_replicas": 0,
            }
        },
    )
    assert (
        k8s_scaling.resolve_scaling_config().active_replicas
        == k8s_scaling.MIN_ACTIVE_REPLICAS
    )


def test_set_deployment_replicas_patches_scale_subresource():
    api = MagicMock()
    k8s_scaling.set_deployment_replicas(api, "my-ns", "my-dep", 0)
    api.patch_namespaced_deployment_scale.assert_called_once_with(
        name="my-dep", namespace="my-ns", body={"spec": {"replicas": 0}}
    )


class _Deployment:
    """Minimal stand-in for V1Deployment's spec.replicas / status.ready_replicas."""

    def __init__(self, desired, ready):
        self.spec = SimpleNamespace(replicas=desired)
        self.status = SimpleNamespace(ready_replicas=ready)


def _api_returning(deployment):
    api = MagicMock()
    api.read_namespaced_deployment.return_value = deployment
    return api


@pytest.mark.parametrize(
    "desired,ready,expected",
    [
        # The prod wedge, 2026-08-14 onward: two pods asked for, none ever Ready
        # (CrashLoopBackOff on an expired Temporal cert). `ready_replicas` comes
        # back as None, not 0 — k8s omits the field rather than zeroing it, and
        # treating None as "unknown, assume healthy" is what would keep the
        # GPUs pinned.
        (2, None, True),
        (2, 0, True),
        (1, None, True),
        # A worker that has Ready pods is doing its job, however busy.
        (2, 1, False),
        (2, 2, False),
        (1, 1, False),
        # Already scaled to zero: nothing is held, so there is nothing to
        # reclaim and this must not read as a wedge (it would make the sweeper
        # claim a scale-down it never performed).
        (0, None, False),
        (0, 0, False),
        (None, None, False),
    ],
)
def test_deployment_is_wedged(desired, ready, expected):
    api = _api_returning(_Deployment(desired, ready))
    assert (
        k8s_scaling.deployment_is_wedged(api, "fishsense", "data-worker") is expected
    )


def test_deployment_is_wedged_tolerates_a_status_less_deployment():
    """A freshly-created Deployment can come back with `status` unset."""
    deployment = SimpleNamespace(spec=SimpleNamespace(replicas=2), status=None)
    api = _api_returning(deployment)
    assert k8s_scaling.deployment_is_wedged(api, "fishsense", "data-worker") is True


def _gpu_config(monkeypatch, **section):
    monkeypatch.setattr(
        k8s_scaling,
        "settings",
        {"kubernetes": {"kubeconfig_path": "/tmp/kc", "namespace": "ns", **section}},
    )
    return k8s_scaling.resolve_scaling_config()


def test_gpu_deployment_names_default_off_the_cpu_name(monkeypatch):
    """Overriding one name must carry the whole trio, or a renamed CPU
    Deployment silently pairs with the stock GPU ones."""
    cfg = _gpu_config(monkeypatch, deployment_name="dw")
    assert cfg.gpu.deployment_name == "dw-gpu"
    assert cfg.gpu.fallback_deployment_name == "dw-gpu-cpu-fallback"


def test_gpu_replica_count_is_independent_of_the_cpu_one(monkeypatch):
    """`active_replicas` sizes the CPU worker; each GPU pod holds a card on a
    contended cluster, so raising CPU throughput must not silently ask NRP for
    more GPUs."""
    cfg = _gpu_config(monkeypatch, active_replicas=4)
    assert cfg.active_replicas == 4
    assert cfg.gpu.policy.active_replicas == 1

    cfg = _gpu_config(monkeypatch, gpu_active_replicas=99)
    assert cfg.gpu.policy.active_replicas == k8s_scaling.MAX_ACTIVE_REPLICAS


def test_fallback_replicas_clamped(monkeypatch):
    cfg = _gpu_config(monkeypatch, gpu_fallback_replicas=99)
    assert cfg.gpu.policy.fallback_replicas == k8s_scaling.MAX_FALLBACK_REPLICAS
    cfg = _gpu_config(monkeypatch, gpu_fallback_replicas=0)
    assert cfg.gpu.policy.fallback_replicas == 1


def test_max_start_failures_is_floored_at_one(monkeypatch):
    """At 0 the pipeline would drop to CPU inference on the first observation
    and never actually try the GPU."""
    assert _gpu_config(monkeypatch, gpu_max_start_failures=0) \
        .gpu.policy.max_start_failures == 1


def test_wedge_grace_cannot_outlast_the_start_timeout(monkeypatch):
    """The activity waits out the start timeout and then observes. A longer
    grace would swallow every observation, no failure would ever be counted,
    and the CPU fallback could never trip."""
    cfg = _gpu_config(
        monkeypatch, gpu_wedge_grace_minutes=60, gpu_start_timeout_seconds=120
    )
    assert cfg.gpu.policy.wedge_grace.total_seconds() == 120


def test_sweep_targets_pair_each_deployment_with_the_queue_it_serves(monkeypatch):
    from fishsense_shared import (
        DATA_PROCESSING_GPU_TASK_QUEUE,
        DATA_PROCESSING_LIGHT_TASK_QUEUE,
        DATA_PROCESSING_TASK_QUEUE,
    )

    cfg = _gpu_config(monkeypatch, deployment_name="dw")
    assert cfg.sweep_targets() == (
        ("dw", DATA_PROCESSING_TASK_QUEUE),
        ("dw-gpu", DATA_PROCESSING_GPU_TASK_QUEUE),
        ("dw-gpu-cpu-fallback", DATA_PROCESSING_GPU_TASK_QUEUE),
        ("dw-light", DATA_PROCESSING_LIGHT_TASK_QUEUE),
    )


def test_the_light_deployment_name_defaults_off_the_cpu_one(monkeypatch):
    """Same rule the GPU names follow: overriding the base name has to carry
    the whole family with it, or a renamed deployment (a test namespace, a
    second environment) is silently paired with the stock light one."""
    cfg = _gpu_config(monkeypatch, deployment_name="dw")
    assert cfg.light.deployment_name == "dw-light"


def test_the_light_worker_can_be_scaled_independently_of_the_cpu_one(monkeypatch):
    """Separate replica knob because the two size against different things.
    `active_replicas` buys rawpy throughput on a memory-bound pod; the light
    worker is bound by neither and one replica serves every stage."""
    cfg = _gpu_config(monkeypatch, active_replicas=3, light_active_replicas=1)
    assert (cfg.active_replicas, cfg.light.active_replicas) == (3, 1)


def test_light_replicas_are_clamped_like_the_cpu_ones(monkeypatch):
    cfg = _gpu_config(monkeypatch, light_active_replicas=99)
    assert cfg.light.active_replicas == k8s_scaling.MAX_ACTIVE_REPLICAS


def test_light_scaling_defaults_to_one_replica(monkeypatch):
    """The light stages are one dive per firing each; a second pod would add
    no throughput and NRP asks us to hold as little as possible."""
    cfg = _gpu_config(monkeypatch)
    assert cfg.light.active_replicas == 1
