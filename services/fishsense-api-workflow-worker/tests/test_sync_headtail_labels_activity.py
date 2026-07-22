# pylint: disable=unused-argument
# Mock side_effect signatures mirror real client positional args; unused
# params are deliberate.
"""Unit tests for sync_headtail_labels_for_label_studio_project_activity.

Mirror of the laser-sync activity tests. Both activities share the
`sync_label_studio_project` helper in `activities/utils.py`; this file
keeps the per-activity regression guards (semaphore + heartbeat) so a
future change to either side is caught locally.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    sync_headtail_labels_for_label_studio_project_activity as sut,
    utils as sut_utils,
)


def _make_task(task_id: int) -> Any:
    return SimpleNamespace(
        id=task_id,
        annotators=[],
        annotations=[],
        is_labeled=False,
        updated_at="2026-05-01T00:00:00Z",
        json=lambda: "{}",
    )


def _make_fs_client(label_lookup, *, cursor=None):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    async def _get(label_studio_id):
        return label_lookup.get(label_studio_id)

    async def _get_cursor(kind, project_id):
        return cursor

    fs.labels = MagicMock()
    fs.labels.get_headtail_label = AsyncMock(side_effect=_get)
    fs.labels.put_headtail_label = AsyncMock()
    fs.labels.get_sync_cursor = AsyncMock(side_effect=_get_cursor)
    fs.labels.put_sync_cursor = AsyncMock()
    return fs


def _make_ls_client(tasks: List[Any]):
    ls = MagicMock()
    ls.projects = MagicMock()
    ls.projects.get = MagicMock(return_value=SimpleNamespace(id=1))
    ls.tasks = MagicMock()
    ls.tasks.list = MagicMock(return_value=tasks)
    return ls


@pytest.mark.asyncio
async def test_per_task_concurrency_is_bounded_by_semaphore(monkeypatch):
    n_tasks = 50
    tasks = [_make_task(i) for i in range(n_tasks)]
    # No annotations + no annotators means put_headtail_label is NOT called
    # in the existing source path. To exercise the semaphore, we need a
    # label that will at least pass through the sem context manager — every
    # task does that regardless of annotations. So count get_headtail_label
    # concurrency instead.
    fs = _make_fs_client(label_lookup={})
    ls = _make_ls_client(tasks)

    in_flight = 0
    peak_in_flight = 0

    async def _slow_get(label_studio_id):
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        in_flight -= 1
        return None

    fs.labels.get_headtail_label = AsyncMock(side_effect=_slow_get)

    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)


    await ActivityEnvironment().run(
        sut.sync_headtail_labels_for_label_studio_project_activity, 1
    )

    assert peak_in_flight <= sut.SYNC_CONCURRENCY, (
        f"peak concurrency was {peak_in_flight}, expected <= {sut.SYNC_CONCURRENCY}"
    )


@pytest.mark.asyncio
async def test_heartbeat_fires_per_completed_task(monkeypatch):
    n_tasks = 5
    tasks = [_make_task(i) for i in range(n_tasks)]

    fs = _make_fs_client(label_lookup={})
    ls = _make_ls_client(tasks)

    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)


    heartbeats: List[tuple] = []

    env = ActivityEnvironment()
    env.on_heartbeat = lambda *args: heartbeats.append(args)

    await env.run(sut.sync_headtail_labels_for_label_studio_project_activity, 1)

    assert len(heartbeats) == n_tasks


# ── Hosted-LS annotator shape ─────────────────────────────────────────
#
# Self-hosted LS returned `task.annotators` as a list of ints. Hosted LS
# (app.heartex.com) returns a list of dicts:
#
#     [{"user_id": 141592, "id": 141592, "username": "ccrutchf", ...}]
#
# The activities passed `annotators[-1]` straight to
# `get_by_label_studio_id`, which URL-encoded the dict into the path
# (`/api/v1/users/label-studio/%7B`) and got a 422. The code anticipated a
# 404, not a 422, so the error escaped the per-task TaskGroup and failed the
# WHOLE project's sync — completions were never written back, and
# fully-labeled projects never fell off the dashboard.

_HOSTED_ANNOTATOR = {
    "user_id": 141592,
    "annotated": True,
    "id": 141592,
    "username": "ccrutchf",
    "email": "ccrutchf@ucsd.edu",
}


def _kp(label: str, x: float, y: float) -> dict:
    return {
        "from_name": "kp-1",
        "original_width": 100,
        "original_height": 200,
        "value": {"x": x, "y": y, "keypointlabels": [label]},
    }


def _labeled_task(task_id: int, annotators) -> Any:
    """A genuinely labeled task: both keypoints placed.

    The activity only writes when the task actually carries annotations, so
    an empty `annotations` list would silently prove nothing here.
    """
    return SimpleNamespace(
        id=task_id,
        annotators=annotators,
        annotations=[{"result": [_kp("Snout", 10, 20), _kp("Fork", 30, 40)]}],
        is_labeled=True,
        updated_at="2026-05-01T00:00:00Z",
        json=lambda: "{}",
    )


@pytest.mark.asyncio
async def test_completion_syncs_when_annotators_are_hosted_ls_dicts(monkeypatch):
    """The regression: a labeled task must reach put_headtail_label."""
    label = SimpleNamespace(
        image_id=7, user_id=None, label_studio_json=None,
        completed=False, updated_at=None, head_x=None, tail_x=None,
    )
    fs = _make_fs_client({1: label})
    fs.users = MagicMock()
    fs.users.get_by_label_studio_id = AsyncMock(
        return_value=SimpleNamespace(id=55)
    )
    ls = _make_ls_client([_labeled_task(1, [_HOSTED_ANNOTATOR])])
    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.sync_headtail_labels_for_label_studio_project_activity, 274633
    )

    # The dict must be unwrapped to its integer id, not passed through.
    fs.users.get_by_label_studio_id.assert_awaited_once_with(141592)
    fs.labels.put_headtail_label.assert_awaited_once()
    written = fs.labels.put_headtail_label.await_args.args[1]
    assert written.completed is True, "completion must reach the DB"
    assert written.user_id == 55


@pytest.mark.asyncio
async def test_annotator_lookup_failure_does_not_lose_the_label(monkeypatch):
    """Attribution is metadata; it must never cost us the completion.

    This is the exact prod failure mode: the lookup raised and took the
    whole project's sync with it.
    """
    label = SimpleNamespace(
        image_id=7, user_id=None, label_studio_json=None,
        completed=False, updated_at=None, head_x=None, tail_x=None,
    )
    fs = _make_fs_client({1: label})
    fs.users = MagicMock()
    fs.users.get_by_label_studio_id = AsyncMock(side_effect=RuntimeError("422"))
    ls = _make_ls_client([_labeled_task(1, [_HOSTED_ANNOTATOR])])
    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.sync_headtail_labels_for_label_studio_project_activity, 274633
    )

    fs.labels.put_headtail_label.assert_awaited_once()
    written = fs.labels.put_headtail_label.await_args.args[1]
    assert written.completed is True
    assert written.user_id is None, "attribution skipped, label still synced"
