# pylint: disable=unused-argument
# Mock side_effect signatures mirror the real client's positional args so
# the AsyncMock substitution is shape-faithful; unused params are deliberate.
"""Unit tests for sync_laser_labels_for_label_studio_project_activity.

Phase 1 regression guards:
  * per-task fan-out is bounded by `SYNC_CONCURRENCY` (was unbounded —
    caused TaskGroup BaseExceptionGroup timeouts in prod).
  * `activity.heartbeat()` fires after each per-task completion so
    Temporal can detect liveness within `heartbeat_timeout`.

Both real clients are mocked; the activity never touches the network.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from label_studio_sdk.core import ApiError
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    sync_laser_labels_for_label_studio_project_activity as sut,
    utils as sut_utils,
)


def _make_task(task_id: int, *, with_annotation: bool = True) -> Any:
    annotation = (
        [
            {
                "result": [
                    {
                        "from_name": "kp-1",
                        "original_width": 1000,
                        "original_height": 800,
                        "value": {"x": 10.0, "y": 20.0, "keypointlabels": ["laser"]},
                    }
                ]
            }
        ]
        if with_annotation
        else []
    )
    return SimpleNamespace(
        id=task_id,
        annotators=[],
        annotations=annotation,
        is_labeled=with_annotation,
        updated_at="2026-05-01T00:00:00Z",
        json=lambda: "{}",
    )


def _make_fs_client(label_lookup, *, cursor=None):
    """Build a mock SDK client whose `labels` sub-client honors a
    per-test mapping of label_studio_id -> stored laser_label and an
    optional starting sync cursor."""
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    async def _get(label_studio_id):
        return label_lookup.get(label_studio_id)

    async def _get_cursor(kind, project_id):
        return cursor

    fs.labels = MagicMock()
    fs.labels.get_laser_label = AsyncMock(side_effect=_get)
    fs.labels.put_laser_label = AsyncMock()
    fs.labels.get_sync_cursor = AsyncMock(side_effect=_get_cursor)
    fs.labels.put_sync_cursor = AsyncMock()
    return fs


def _make_ls_client(tasks: List[Any], *, project_exists: bool = True):
    ls = MagicMock()
    ls.projects = MagicMock()
    if project_exists:
        ls.projects.get = MagicMock(return_value=SimpleNamespace(id=1))
    else:
        ls.projects.get = MagicMock(side_effect=ApiError(status_code=404, body="missing"))
    ls.tasks = MagicMock()
    ls.tasks.list = MagicMock(return_value=tasks)
    return ls


@pytest.mark.asyncio
async def test_skips_tasks_with_no_existing_label(monkeypatch):
    tasks = [_make_task(i) for i in range(3)]
    fs = _make_fs_client(label_lookup={})  # no labels stored — every get returns None
    ls = _make_ls_client(tasks)

    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)


    await ActivityEnvironment().run(
        sut.sync_laser_labels_for_label_studio_project_activity, 1
    )

    assert fs.labels.get_laser_label.await_count == 3
    assert fs.labels.put_laser_label.await_count == 0


@pytest.mark.asyncio
async def test_returns_early_when_project_missing(monkeypatch):
    fs = _make_fs_client(label_lookup={})
    ls = _make_ls_client([], project_exists=False)

    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)


    await ActivityEnvironment().run(
        sut.sync_laser_labels_for_label_studio_project_activity, 999
    )

    ls.tasks.list.assert_not_called()


@pytest.mark.asyncio
async def test_per_task_concurrency_is_bounded_by_semaphore(monkeypatch):
    """The Phase 1 regression guard: with N >> SYNC_CONCURRENCY tasks,
    the activity must never have more than SYNC_CONCURRENCY in-flight
    `put_laser_label` calls. Pre-fix this was unbounded."""
    n_tasks = 50
    tasks = [_make_task(i) for i in range(n_tasks)]
    label_lookup = {i: SimpleNamespace(image_id=i, label_studio_json={}) for i in range(n_tasks)}

    fs = _make_fs_client(label_lookup=label_lookup)
    ls = _make_ls_client(tasks)

    in_flight = 0
    peak_in_flight = 0

    async def _slow_put(image_id, label):
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        # Yield to the event loop so other coroutines have a chance to
        # contend for the semaphore. If the sem isn't enforced, peak
        # will jump straight to n_tasks.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        in_flight -= 1

    fs.labels.put_laser_label = AsyncMock(side_effect=_slow_put)

    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)


    await ActivityEnvironment().run(
        sut.sync_laser_labels_for_label_studio_project_activity, 1
    )

    assert peak_in_flight <= sut.SYNC_CONCURRENCY, (
        f"peak concurrency was {peak_in_flight}, expected <= {sut.SYNC_CONCURRENCY}"
    )
    assert fs.labels.put_laser_label.await_count == n_tasks


@pytest.mark.asyncio
async def test_heartbeat_fires_per_completed_task(monkeypatch):
    n_tasks = 5
    tasks = [_make_task(i) for i in range(n_tasks)]
    label_lookup = {i: SimpleNamespace(image_id=i, label_studio_json={}) for i in range(n_tasks)}

    fs = _make_fs_client(label_lookup=label_lookup)
    ls = _make_ls_client(tasks)

    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)


    heartbeats: List[tuple] = []

    env = ActivityEnvironment()
    env.on_heartbeat = lambda *args: heartbeats.append(args)

    await env.run(sut.sync_laser_labels_for_label_studio_project_activity, 1)

    assert len(heartbeats) == n_tasks
