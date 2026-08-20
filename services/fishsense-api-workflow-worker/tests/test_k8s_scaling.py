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
