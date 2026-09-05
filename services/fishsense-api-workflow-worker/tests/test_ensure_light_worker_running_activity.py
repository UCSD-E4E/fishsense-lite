"""The light-queue worker's wake activity.

Mirrors `test_ensure_data_worker_running_activity` — the two differ only in
which Deployment and which replica knob they read, and that difference is the
whole point: the light worker must not be sized by `active_replicas`, which
exists to buy rawpy throughput on the memory-bound per-image pod.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fishsense_api_workflow_worker.activities import (
    ensure_light_worker_running_activity as sut,
)
from fishsense_api_workflow_worker.activities.k8s_scaling import (
    LightScalingConfig,
    ScalingConfig,
)


def _config(**light) -> ScalingConfig:
    return ScalingConfig(
        kubeconfig_path="/tmp/kc",
        namespace="ns",
        deployment_name="dw",
        active_replicas=3,
        idle_cooldown_minutes=15,
        light=LightScalingConfig(**light),
    )


@pytest.mark.asyncio
async def test_it_scales_the_light_deployment_not_the_per_image_one(monkeypatch):
    """The bug this guards: reading `deployment_name` / `active_replicas` here
    would scale the rawpy worker and leave the light queue unserved, so every
    light child would hang until its schedule-to-close."""
    scaled = MagicMock()
    monkeypatch.setattr(
        sut, "resolve_scaling_config",
        lambda: _config(deployment_name="dw-light", active_replicas=1),
    )
    monkeypatch.setattr(sut, "apps_v1_api", lambda _path: "api")
    monkeypatch.setattr(sut, "set_deployment_replicas", scaled)

    result = await sut.ensure_light_worker_running_activity()

    assert result == 1
    scaled.assert_called_once_with("api", "ns", "dw-light", 1)


@pytest.mark.asyncio
async def test_it_is_a_no_op_when_scaling_is_not_configured(monkeypatch):
    """Local devcontainer and pre-NRP: the worker runs `role = "all"` and is
    always up, so waking it is meaningless rather than an error."""
    scaled = MagicMock()
    monkeypatch.setattr(sut, "resolve_scaling_config", lambda: None)
    monkeypatch.setattr(sut, "set_deployment_replicas", scaled)

    assert await sut.ensure_light_worker_running_activity() == 0
    scaled.assert_not_called()
