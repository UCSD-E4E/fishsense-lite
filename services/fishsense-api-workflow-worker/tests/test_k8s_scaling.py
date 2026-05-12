"""Unit tests for the shared NRP data-worker scaling helpers.

These don't touch Kubernetes or Temporal — they pin the config
resolution (disabled-by-default, required namespace, clamped replica
count, defaults) and the declarative replica-set call shape.
"""

from __future__ import annotations

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
