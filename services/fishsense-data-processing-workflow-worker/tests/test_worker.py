"""Worker wiring tests.

The data-worker can be scaled to zero on NRP and woken on demand, so a
scale-down sends SIGTERM mid-activity. These tests pin that the Worker
is constructed with a non-trivial ``graceful_shutdown_timeout`` (so
in-flight rectify/measure activities get a chance to finish before
they're cancelled and re-queued) and that ``build_worker`` is the
single construction point used by ``main``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from temporalio.testing import WorkflowEnvironment

from fishsense_data_processing_workflow_worker.worker import (
    DEFAULT_MAX_CONCURRENT_ACTIVITIES,
    GRACEFUL_SHUTDOWN_TIMEOUT,
    TASK_QUEUE_NAME,
    build_worker,
)


def test_graceful_shutdown_timeout_is_a_positive_duration():
    assert isinstance(GRACEFUL_SHUTDOWN_TIMEOUT, timedelta)
    assert GRACEFUL_SHUTDOWN_TIMEOUT > timedelta(0)


@pytest.mark.asyncio
async def test_build_worker_wires_graceful_shutdown_and_task_queue():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        with ThreadPoolExecutor(max_workers=1) as executor:
            worker = build_worker(env.client, executor)
            config = worker.config()
            assert config["task_queue"] == TASK_QUEUE_NAME
            assert config["graceful_shutdown_timeout"] == GRACEFUL_SHUTDOWN_TIMEOUT
            # Default concurrency cap bounds peak memory (rawpy OOM guard).
            assert (
                config["max_concurrent_activities"]
                == DEFAULT_MAX_CONCURRENT_ACTIVITIES
            )


@pytest.mark.asyncio
async def test_build_worker_honors_explicit_concurrency_cap():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        with ThreadPoolExecutor(max_workers=1) as executor:
            worker = build_worker(env.client, executor, max_concurrent_activities=2)
            assert worker.config()["max_concurrent_activities"] == 2
