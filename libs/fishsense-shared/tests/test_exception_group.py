"""Tests for ExceptionGroupErrorLogging.

The helper has two callsite shapes:

1. Activity / startup contexts (``suppress=False``, the default) where
   any task-group failure should still propagate so Temporal converts
   it to an activity / startup failure as usual.
2. Workflow contexts that explicitly want fire-and-forget semantics —
   e.g. the post-sync validation pass in
   ``SyncLabelStudioLaserLabelsWorkflow``, which must not roll back a
   successful sync if a downstream child workflow fails. Without
   ``suppress=True`` the bare ``BaseExceptionGroup`` raised by
   ``asyncio.TaskGroup`` is not a ``temporalio.exceptions.FailureError``
   subclass, so Temporal treats it as
   ``WORKFLOW_TASK_FAILED_CAUSE_WORKFLOW_WORKER_UNHANDLED_FAILURE`` and
   retries the workflow task forever.
"""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import asyncio
import logging

import pytest

from fishsense_shared import ExceptionGroupErrorLogging


@pytest.fixture
def caplog_at_error(caplog):
    caplog.set_level(logging.ERROR)
    return caplog


@pytest.mark.asyncio
async def test_clean_exit_does_not_log(caplog_at_error):
    logger = logging.getLogger("test_clean_exit")
    async with ExceptionGroupErrorLogging(logger):
        pass
    assert not caplog_at_error.records


@pytest.mark.asyncio
async def test_propagates_exception_group_by_default(caplog_at_error):
    logger = logging.getLogger("test_propagate_group")

    group = BaseExceptionGroup(
        "from taskgroup",
        [ValueError("a"), RuntimeError("b")],
    )

    with pytest.raises(BaseExceptionGroup) as exc_info:
        async with ExceptionGroupErrorLogging(logger):
            raise group

    assert exc_info.value is group
    sub_records = [r for r in caplog_at_error.records if "Sub-exception" in r.message]
    assert len(sub_records) == 2


@pytest.mark.asyncio
async def test_propagates_bare_exception_by_default(caplog_at_error):
    logger = logging.getLogger("test_propagate_bare")

    with pytest.raises(ValueError):
        async with ExceptionGroupErrorLogging(logger):
            raise ValueError("solo")

    assert any("Error in TaskGroup" in r.message for r in caplog_at_error.records)


@pytest.mark.asyncio
async def test_suppresses_exception_group_when_suppress_true(caplog_at_error):
    logger = logging.getLogger("test_suppress_group")

    group = BaseExceptionGroup(
        "from taskgroup",
        [ValueError("a"), RuntimeError("b"), KeyError("c")],
    )

    async with ExceptionGroupErrorLogging(logger, suppress=True):
        raise group

    sub_records = [r for r in caplog_at_error.records if "Sub-exception" in r.message]
    assert len(sub_records) == 3


@pytest.mark.asyncio
async def test_suppresses_bare_exception_when_suppress_true(caplog_at_error):
    logger = logging.getLogger("test_suppress_bare")

    async with ExceptionGroupErrorLogging(logger, suppress=True):
        raise ValueError("solo")

    assert any("Error in TaskGroup" in r.message for r in caplog_at_error.records)


@pytest.mark.asyncio
async def test_real_taskgroup_propagates_by_default(caplog_at_error):
    """Integration check against a real ``asyncio.TaskGroup``: 3 failing
    children → 3 logged sub-exceptions and the group still propagates so
    upstream callers can fail loudly when they want to."""
    logger = logging.getLogger("test_real_taskgroup_propagate")

    async def boom(label: str) -> None:
        raise RuntimeError(label)

    with pytest.raises(BaseExceptionGroup) as exc_info:
        async with ExceptionGroupErrorLogging(logger):
            async with asyncio.TaskGroup() as tg:
                tg.create_task(boom("a"))
                tg.create_task(boom("b"))
                tg.create_task(boom("c"))

    assert len(exc_info.value.exceptions) == 3
    sub_records = [r for r in caplog_at_error.records if "Sub-exception" in r.message]
    assert len(sub_records) == 3


@pytest.mark.asyncio
async def test_real_taskgroup_swallowed_when_suppress_true(caplog_at_error):
    """The validation-pass scenario from the failing workflow event:
    multiple child workflow dispatches blow up, the helper logs each,
    and the surrounding workflow continues."""
    logger = logging.getLogger("test_real_taskgroup_suppress")

    async def boom(label: str) -> None:
        raise RuntimeError(label)

    after_block_ran = False
    async with ExceptionGroupErrorLogging(logger, suppress=True):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(boom("a"))
            tg.create_task(boom("b"))
    after_block_ran = True

    assert after_block_ran
    sub_records = [r for r in caplog_at_error.records if "Sub-exception" in r.message]
    assert len(sub_records) == 2
