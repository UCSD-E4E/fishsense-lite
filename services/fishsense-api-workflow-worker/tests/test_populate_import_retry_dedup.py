"""The import path must never create a second copy of a task.

Ten Label Studio projects were found on 2026-09-09 holding two tasks per
image, and one held 23 copies of three images. `import_tasks_and_record_labels`
made them itself: hosted LS imports asynchronously, the helper polled ~24s for
the new tasks to become listable, and **raised** when they had not appeared.
The populate activities declare no `RetryPolicy`, so Temporal retried inside
the 30-minute `schedule_to_close`, the retry's dedup listing still could not
see the in-flight import, and it imported the whole set again. Both copies then
materialised. The raise even said "retrying (dedupe prevents dupes)" — which
holds only when the previous import has already materialised, precisely the
condition that just failed.

The consequences went past wasted labeller queue: a duplicate splits a row from
its own annotation (seven dive-437 head/tail labels were invisible to the
pipeline because the row tracked the empty twin), and it hands a labeller a
second, *ungated* copy of a frame the laser auto-accept gate had already
judged — all 66 of dive 517's calibration frames were re-done by hand that way.

So the contract these tests pin is: **an import that has not become visible is
not an error.** Write rows for what is visible, leave the rest to the next
scheduled run, and never let a retry re-issue the import.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock

import pytest
from label_studio_sdk.core import ApiError
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import populate_utils as sut


def _resolve_url(task_id: int, s3_uri: str) -> str:
    """The presign wrapper hosted LS serves back instead of the s3:// URI."""
    fileuri = base64.b64encode(s3_uri.encode()).decode()
    return f"/tasks/{task_id}/resolve/?fileuri={fileuri}"


class SlowImportLabelStudio:
    """Hosted LS whose imported tasks only become listable after a delay.

    `visible_after_lists` is how many `tasks.list` calls must go by before the
    tasks an import created show up. That is the real behaviour the helper has
    to survive; the shared fake in `worker_tests_support.populate` makes them
    visible immediately, which is why it cannot express this bug.
    """

    def __init__(self, task_ids: List[int], *, visible_after_lists: int = 0):
        self._ids = iter(task_ids)
        self._visible: List[SimpleNamespace] = []
        self._pending: List[tuple[int, SimpleNamespace]] = []
        self._lists = 0
        self.import_calls: List[List[dict]] = []
        self.list_raises: List[Exception] = []
        self._visible_after = visible_after_lists
        self.projects = MagicMock()
        self.projects.import_tasks = MagicMock(side_effect=self._import)
        self.tasks = MagicMock()
        self.tasks.list = MagicMock(side_effect=self._list)

    def _import(self, project_id, request, return_task_ids=False):
        # pylint: disable=unused-argument
        self.import_calls.append(list(request))
        for task in request:
            task_id = next(self._ids)
            s3_uri = task["data"].get("image") or task["data"].get("img")
            self._pending.append(
                (
                    self._lists + self._visible_after,
                    SimpleNamespace(
                        id=task_id, data={"image": _resolve_url(task_id, s3_uri)}
                    ),
                )
            )
        return SimpleNamespace(import_=1)

    def _list(self, project=None):  # pylint: disable=unused-argument
        if self.list_raises:
            raise self.list_raises.pop(0)
        due = [t for due_at, t in self._pending if due_at <= self._lists]
        self._pending = [p for p in self._pending if p[0] > self._lists]
        self._visible.extend(due)
        self._lists += 1
        return list(self._visible)

    def reveal_all(self):
        """Let every still-pending task become listable on the next list."""
        self._visible.extend(t for _, t in self._pending)
        self._pending = []


def _task(checksum: str) -> dict:
    return {"data": {"image": f"s3://bucket/preprocess_jpeg/{checksum}.JPG"}}


@pytest.fixture(name="fast_poll")
def _fast_poll(monkeypatch):
    """Keep the visibility budget short and sleep-free."""
    monkeypatch.setattr(sut, "_IMPORT_VISIBILITY_ATTEMPTS", 2)
    monkeypatch.setattr(sut, "_IMPORT_VISIBILITY_INTERVAL_S", 0)


async def _run(ls, tasks, items, recorded, project_id=901):
    async def record_label(item, task_id):
        recorded.append((item, task_id))

    async def call():
        return await sut.import_tasks_and_record_labels(
            project_id=project_id,
            tasks=tasks,
            record_label=record_label,
            items=items,
        )

    return await ActivityEnvironment().run(call)


@pytest.mark.usefixtures("fast_poll")
async def test_invisible_import_does_not_raise_and_is_not_reimported(monkeypatch):
    """The whole bug, end to end.

    An import that has not materialised inside the budget must not raise —
    raising is what made Temporal retry, and the retry is what duplicated. The
    next run must then reconcile the tasks rather than import them again.
    """
    ls = SlowImportLabelStudio([11, 12], visible_after_lists=99)
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)
    tasks = [_task("aaa"), _task("bbb")]

    recorded: List[tuple] = []
    written = await _run(ls, tasks, ["a", "b"], recorded)

    assert len(ls.import_calls) == 1, "the batch is imported exactly once"
    assert written == 0 and recorded == [], "nothing visible yet, so no rows"

    # The next scheduled populate run, by which time LS has caught up.
    ls.reveal_all()
    recorded.clear()
    written = await _run(ls, tasks, ["a", "b"], recorded)

    assert len(ls.import_calls) == 1, "the second run must NOT re-import"
    assert written == 2
    assert sorted(recorded) == [("a", 11), ("b", 12)]


@pytest.mark.usefixtures("fast_poll")
async def test_partial_visibility_records_what_landed(monkeypatch):
    """A task that did land gets its row now, not an hour later."""
    ls = SlowImportLabelStudio([21, 22], visible_after_lists=0)
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)
    ls._pending = []  # noqa: SLF001 - seed one task as pre-existing
    ls._visible = [  # noqa: SLF001
        SimpleNamespace(
            id=99, data={"image": _resolve_url(99, "s3://bucket/p/aaa.JPG")}
        )
    ]

    recorded: List[tuple] = []
    written = await _run(
        ls,
        [{"data": {"image": "s3://bucket/p/aaa.JPG"}}, _task("bbb")],
        ["a", "b"],
        recorded,
    )

    assert ls.import_calls == [[_task("bbb")]], "only the missing task imported"
    assert written == 2
    assert sorted(recorded) == [("a", 99), ("b", 21)]


@pytest.mark.usefixtures("fast_poll")
async def test_repeated_url_in_one_batch_is_imported_once(monkeypatch):
    """Dedup has to cover the input batch, not just the project.

    The helper only ever compared against tasks *already in the project*, so a
    caller whose item query returned an image twice imported it twice in a
    single call, with nothing in the project yet to compare against.
    """
    ls = SlowImportLabelStudio([31])
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)
    dup = _task("same")

    recorded: List[tuple] = []
    written = await _run(ls, [dup, dict(dup)], ["first", "second"], recorded)

    assert len(ls.import_calls[0]) == 1, "the repeated URL is imported once"
    assert written == 2
    assert sorted(recorded) == [("first", 31), ("second", 31)]


@pytest.mark.usefixtures("fast_poll")
async def test_throttled_listing_is_retried_not_propagated(monkeypatch):
    """A 429 is the other route into the duplicating retry.

    `populate_utils` had no throttle handling at all, unlike the sync path
    beside it, so a rate limit failed the activity and Temporal retried it —
    the same race, reached a different way.
    """
    monkeypatch.setattr(sut, "_throttle_sleep", _no_sleep)
    ls = SlowImportLabelStudio([41])
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)
    ls.list_raises = [
        ApiError(status_code=429, body={"detail": "Request was throttled."})
    ]

    recorded: List[tuple] = []
    written = await _run(ls, [_task("ccc")], ["c"], recorded)

    assert written == 1
    assert recorded == [("c", 41)]


async def _no_sleep(_seconds):
    return None
