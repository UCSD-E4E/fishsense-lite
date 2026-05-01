# pylint: disable=unused-argument
# Mock side_effect / update_fn signatures mirror real client positional
# args; unused params are deliberate.
"""Phase 4 regression guards for incremental sync.

Asserts the helper's contract with `LabelStudioSyncCursor`:
  * with no cursor, every task is processed and a cursor is written at
    `max(task.updated_at)`.
  * with a cursor, only strictly-newer tasks are processed and the
    cursor advances to the new max.
  * on per-task failure, the cursor is NOT advanced (replay-safe via
    upsert PUTs).
  * `kind` is forwarded to both get/put cursor calls so laser and
    headtail keep separate cursors per project.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio import activity
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import utils as sut_utils


def _task(task_id: int, *, updated_at: str) -> Any:
    return SimpleNamespace(
        id=task_id,
        annotators=[],
        annotations=[],
        is_labeled=False,
        updated_at=updated_at,
        json=lambda: "{}",
    )


def _make_fs_client(*, cursor=None):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    async def _get_cursor(kind, project_id):
        return cursor

    fs.labels = MagicMock()
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
async def test_no_cursor_processes_every_task_and_writes_max_seen(monkeypatch):
    tasks = [
        _task(1, updated_at="2026-04-01T00:00:00Z"),
        _task(2, updated_at="2026-04-15T00:00:00Z"),
        _task(3, updated_at="2026-04-10T00:00:00Z"),
    ]

    processed: List[int] = []

    async def update_fn(_fs, task):
        processed.append(task.id)

    fs = _make_fs_client(cursor=None)
    ls = _make_ls_client(tasks)
    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)


    async def _run():
        await sut_utils.sync_label_studio_project(42, update_fn, kind="laser")


    @activity.defn(name="_test")
    async def stub_activity():
        await _run()

    await ActivityEnvironment().run(stub_activity)

    assert sorted(processed) == [1, 2, 3]
    fs.labels.put_sync_cursor.assert_awaited_once()
    args, _ = fs.labels.put_sync_cursor.await_args
    kind, project_id, written_cursor = args
    assert kind == "laser"
    assert project_id == 42
    assert written_cursor.last_synced_at == datetime(
        2026, 4, 15, 0, 0, 0, tzinfo=timezone.utc
    )


@pytest.mark.asyncio
async def test_cursor_filters_out_tasks_at_or_before_high_water(monkeypatch):
    tasks = [
        _task(1, updated_at="2026-04-01T00:00:00Z"),
        _task(2, updated_at="2026-04-10T00:00:00Z"),  # == cursor: skip
        _task(3, updated_at="2026-04-12T00:00:00Z"),
        _task(4, updated_at="2026-04-09T00:00:00Z"),  # < cursor: skip
    ]
    cursor = SimpleNamespace(
        id=7,
        kind="laser",
        label_studio_project_id=42,
        last_synced_at=datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc),
    )

    processed: List[int] = []

    async def update_fn(_fs, task):
        processed.append(task.id)

    fs = _make_fs_client(cursor=cursor)
    ls = _make_ls_client(tasks)
    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)


    @activity.defn(name="_test")
    async def stub_activity():
        await sut_utils.sync_label_studio_project(42, update_fn, kind="laser")

    await ActivityEnvironment().run(stub_activity)

    assert processed == [3]
    args, _ = fs.labels.put_sync_cursor.await_args
    _, _, written_cursor = args
    assert written_cursor.last_synced_at == datetime(
        2026, 4, 12, 0, 0, 0, tzinfo=timezone.utc
    )
    # Existing cursor id is reused so the row is updated, not duplicated.
    assert written_cursor.id == 7


@pytest.mark.asyncio
async def test_cursor_not_written_when_no_new_tasks(monkeypatch):
    tasks = [_task(1, updated_at="2026-04-01T00:00:00Z")]
    cursor = SimpleNamespace(
        id=7,
        kind="laser",
        label_studio_project_id=42,
        last_synced_at=datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc),
    )

    async def update_fn(_fs, task):
        raise AssertionError("update_fn should not be invoked")

    fs = _make_fs_client(cursor=cursor)
    ls = _make_ls_client(tasks)
    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)


    @activity.defn(name="_test")
    async def stub_activity():
        await sut_utils.sync_label_studio_project(42, update_fn, kind="laser")

    await ActivityEnvironment().run(stub_activity)

    fs.labels.put_sync_cursor.assert_not_called()


@pytest.mark.asyncio
async def test_cursor_not_advanced_on_per_task_failure(monkeypatch):
    tasks = [
        _task(1, updated_at="2026-04-01T00:00:00Z"),
        _task(2, updated_at="2026-04-15T00:00:00Z"),
    ]

    async def update_fn(_fs, task):
        if task.id == 2:
            raise RuntimeError("simulated downstream failure")

    fs = _make_fs_client(cursor=None)
    ls = _make_ls_client(tasks)
    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)


    @activity.defn(name="_test")
    async def stub_activity():
        await sut_utils.sync_label_studio_project(42, update_fn, kind="laser")

    with pytest.raises(BaseException):  # ExceptionGroup or RuntimeError
        await ActivityEnvironment().run(stub_activity)

    fs.labels.put_sync_cursor.assert_not_called()


@pytest.mark.asyncio
async def test_kind_is_forwarded_to_cursor_calls(monkeypatch):
    tasks = [_task(1, updated_at="2026-04-01T00:00:00Z")]

    async def update_fn(_fs, task):
        return None

    fs = _make_fs_client(cursor=None)
    ls = _make_ls_client(tasks)
    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)


    @activity.defn(name="_test")
    async def stub_activity():
        await sut_utils.sync_label_studio_project(42, update_fn, kind="headtail")

    await ActivityEnvironment().run(stub_activity)

    fs.labels.get_sync_cursor.assert_awaited_once_with("headtail", 42)
    args, _ = fs.labels.put_sync_cursor.await_args
    assert args[0] == "headtail"
    assert args[1] == 42


@pytest.mark.asyncio
async def test_heartbeat_pumps_during_slow_ls_listing(monkeypatch):
    """A backlog project's pager iterates synchronously inside a worker
    thread; the helper must pump heartbeats from the asyncio main thread
    so the 2m heartbeat_timeout doesn't trip mid-listing.
    """
    monkeypatch.setattr(sut_utils, "_LISTING_HEARTBEAT_INTERVAL_SECONDS", 0.05)

    tasks = [_task(1, updated_at="2026-04-01T00:00:00Z")]

    def _slow_list(*_args, **_kwargs):
        # Block the worker thread long enough for ≥2 heartbeat ticks.
        time.sleep(0.25)
        return tasks

    fs = _make_fs_client(cursor=None)
    ls = _make_ls_client(tasks)
    ls.tasks.list = MagicMock(side_effect=_slow_list)
    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)

    heartbeats: list[tuple] = []

    @activity.defn(name="_test")
    async def stub_activity():
        await sut_utils.sync_label_studio_project(42, lambda *_: _noop(), kind="laser")

    async def _noop():
        return None

    env = ActivityEnvironment()
    env.on_heartbeat = lambda *args: heartbeats.append(args)
    await env.run(stub_activity)

    # ≥2 from the listing pump (per-task heartbeat in _run is one more).
    assert len(heartbeats) >= 2
