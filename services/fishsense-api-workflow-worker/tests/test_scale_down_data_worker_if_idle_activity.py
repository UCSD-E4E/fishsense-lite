# pylint: disable=unnecessary-lambda,protected-access
"""Unit tests for scale_down_data_worker_if_idle_activity.

The hourly sweeper scales the NRP data-worker Deployment to 0 only
when its task queue has had no running or recently-closed workflow.
The Temporal-busy check and the k8s client are mocked; this pins:
disabled → no-op; busy → leave replicas; idle → scale to 0; and the
shape of the Temporal list-filter query.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    scale_down_data_worker_if_idle_activity as sut,
)
from fishsense_api_workflow_worker.activities.k8s_scaling import ScalingConfig


def _config(cooldown: int = 15) -> ScalingConfig:
    return ScalingConfig(
        kubeconfig_path="/tmp/nrp.kubeconfig",
        namespace="fishsense",
        deployment_name="fishsense-data-processing-workflow-worker",
        active_replicas=1,
        idle_cooldown_minutes=cooldown,
    )


def _stub_busy(monkeypatch, value: bool) -> None:
    async def _busy(_cooldown: int) -> bool:
        return value

    monkeypatch.setattr(sut, "_data_worker_task_queue_busy", _busy)


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
async def test_does_not_scale_down_when_busy(monkeypatch):
    monkeypatch.setattr(sut, "resolve_scaling_config", lambda: _config())
    _stub_busy(monkeypatch, True)
    monkeypatch.setattr(
        sut,
        "set_deployment_replicas",
        lambda *_a, **_k: pytest.fail("must not scale a busy data-worker"),
    )
    result = await ActivityEnvironment().run(
        sut.scale_down_data_worker_if_idle_activity
    )
    assert result is False


@pytest.mark.asyncio
async def test_scales_to_zero_when_idle(monkeypatch):
    monkeypatch.setattr(sut, "resolve_scaling_config", lambda: _config())
    _stub_busy(monkeypatch, False)
    api = MagicMock()
    monkeypatch.setattr(sut, "apps_v1_api", lambda path: api)
    calls: list = []
    monkeypatch.setattr(
        sut, "set_deployment_replicas", lambda a, ns, name, n: calls.append((ns, name, n))
    )
    result = await ActivityEnvironment().run(
        sut.scale_down_data_worker_if_idle_activity
    )
    assert result is True
    assert calls == [("fishsense", "fishsense-data-processing-workflow-worker", 0)]


def test_busy_query_targets_data_worker_task_queue_with_running_or_recent_close():
    query = sut._build_busy_query(15)
    assert 'TaskQueue = "fishsense_data_processing_queue"' in query
    assert 'ExecutionStatus = "Running"' in query
    # A recent-close cutoff timestamp (RFC3339, Z-suffixed) is present.
    assert 'CloseTime > "20' in query and query.rstrip().endswith('Z")')
