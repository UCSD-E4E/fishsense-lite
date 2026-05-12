"""Unit tests for ensure_data_worker_running_activity.

The activity scales the NRP data-worker Deployment up to the
configured ``active_replicas`` before a parent dispatches a child
workflow — or no-ops (returns 0) when k8s scaling isn't configured.
The k8s client and config resolution are mocked; this pins the
behavior, not the wire calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    ensure_data_worker_running_activity as sut,
)
from fishsense_api_workflow_worker.activities.k8s_scaling import ScalingConfig


def _config(active_replicas: int = 1) -> ScalingConfig:
    return ScalingConfig(
        kubeconfig_path="/tmp/nrp.kubeconfig",
        namespace="fishsense",
        deployment_name="fishsense-data-processing-workflow-worker",
        active_replicas=active_replicas,
        idle_cooldown_minutes=15,
    )


@pytest.mark.asyncio
async def test_noop_returns_zero_when_scaling_disabled(monkeypatch):
    monkeypatch.setattr(sut, "resolve_scaling_config", lambda: None)
    # If it tried to touch k8s with no config it would blow up here.
    monkeypatch.setattr(
        sut, "apps_v1_api", lambda *_a, **_k: pytest.fail("must not build k8s client")
    )

    result = await ActivityEnvironment().run(
        sut.ensure_data_worker_running_activity
    )
    assert result == 0


@pytest.mark.asyncio
async def test_scales_up_to_active_replicas(monkeypatch):
    monkeypatch.setattr(sut, "resolve_scaling_config", lambda: _config(active_replicas=2))
    api = MagicMock()
    monkeypatch.setattr(sut, "apps_v1_api", lambda path: api)
    calls: list = []
    monkeypatch.setattr(
        sut,
        "set_deployment_replicas",
        lambda a, ns, name, n: calls.append((a is api, ns, name, n)),
    )

    result = await ActivityEnvironment().run(
        sut.ensure_data_worker_running_activity
    )

    assert result == 2
    assert calls == [(True, "fishsense", "fishsense-data-processing-workflow-worker", 2)]


@pytest.mark.asyncio
async def test_default_single_replica(monkeypatch):
    monkeypatch.setattr(sut, "resolve_scaling_config", lambda: _config())
    monkeypatch.setattr(sut, "apps_v1_api", lambda path: MagicMock())
    targets: list = []
    monkeypatch.setattr(
        sut, "set_deployment_replicas", lambda a, ns, name, n: targets.append(n)
    )

    result = await ActivityEnvironment().run(
        sut.ensure_data_worker_running_activity
    )
    assert result == 1
    assert targets == [1]
