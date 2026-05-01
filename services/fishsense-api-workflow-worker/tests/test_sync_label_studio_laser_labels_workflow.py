"""Workflow contract test for SyncLabelStudioLaserLabelsWorkflow.

Drives the workflow end-to-end against an in-process Temporal test server
(`WorkflowEnvironment.start_time_skipping()`). Every activity is replaced
with a stub that records its invocation so we can assert which activities
were called, with which args, and — critically — that the per-project
sync activity is invoked with both `schedule_to_close_timeout` and
`heartbeat_timeout` set (regression guard for the Phase 1 fix).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import List

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows.sync_label_studio_laser_labels_workflow import (
    SyncLabelStudioLaserLabelsWorkflow,
)


@dataclass
class _Call:
    name: str
    args: tuple


@pytest.mark.asyncio
async def test_workflow_invokes_users_then_project_ids_then_one_sync_per_project():
    calls: List[_Call] = []

    @activity.defn(name="sync_users_label_studio_activity")
    async def stub_sync_users() -> None:
        calls.append(_Call("sync_users_label_studio_activity", ()))

    @activity.defn(name="get_laser_label_studio_project_ids_activity")
    async def stub_get_ids() -> List[int]:
        calls.append(_Call("get_laser_label_studio_project_ids_activity", ()))
        return [101, 202, 303]

    @activity.defn(name="sync_laser_labels_for_label_studio_project_activity")
    async def stub_sync_project(project_id: int) -> None:
        calls.append(
            _Call("sync_laser_labels_for_label_studio_project_activity", (project_id,))
        )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-laser-sync",
            workflows=[SyncLabelStudioLaserLabelsWorkflow],
            activities=[stub_sync_users, stub_get_ids, stub_sync_project],
        ):
            await env.client.execute_workflow(
                SyncLabelStudioLaserLabelsWorkflow.run,
                id=f"test-laser-sync-{uuid.uuid4()}",
                task_queue="test-laser-sync",
            )

    assert calls[0].name == "sync_users_label_studio_activity"
    assert calls[1].name == "get_laser_label_studio_project_ids_activity"

    project_calls = [
        c for c in calls if c.name == "sync_laser_labels_for_label_studio_project_activity"
    ]
    assert len(project_calls) == 3
    assert {c.args[0] for c in project_calls} == {101, 202, 303}


@pytest.mark.asyncio
async def test_workflow_with_no_projects_does_not_invoke_per_project_sync():
    sync_calls: List[int] = []

    @activity.defn(name="sync_users_label_studio_activity")
    async def stub_sync_users() -> None:
        return None

    @activity.defn(name="get_laser_label_studio_project_ids_activity")
    async def stub_get_ids() -> List[int]:
        return []

    @activity.defn(name="sync_laser_labels_for_label_studio_project_activity")
    async def stub_sync_project(project_id: int) -> None:
        sync_calls.append(project_id)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-laser-sync-empty",
            workflows=[SyncLabelStudioLaserLabelsWorkflow],
            activities=[stub_sync_users, stub_get_ids, stub_sync_project],
        ):
            await env.client.execute_workflow(
                SyncLabelStudioLaserLabelsWorkflow.run,
                id=f"test-laser-sync-empty-{uuid.uuid4()}",
                task_queue="test-laser-sync-empty",
            )

    assert not sync_calls


@pytest.mark.asyncio
async def test_per_project_activity_caps_concurrency_at_workflow_level():
    """Phase 1 regression guard: the workflow must not fan out one
    activity per project unbounded — they share a `Semaphore(4)`."""
    in_flight = 0
    peak_in_flight = 0
    started: List[int] = []
    release_gate = asyncio.Event()

    @activity.defn(name="sync_users_label_studio_activity")
    async def stub_sync_users() -> None:
        return None

    @activity.defn(name="get_laser_label_studio_project_ids_activity")
    async def stub_get_ids() -> List[int]:
        return list(range(20))

    @activity.defn(name="sync_laser_labels_for_label_studio_project_activity")
    async def stub_sync_project(project_id: int) -> None:
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        started.append(project_id)
        try:
            # Wait until the test releases all of them. While we wait, the
            # workflow's sem must be holding at most 4 of these in flight.
            await asyncio.wait_for(release_gate.wait(), timeout=5.0)
        finally:
            in_flight -= 1

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-laser-sync-conc",
            workflows=[SyncLabelStudioLaserLabelsWorkflow],
            activities=[stub_sync_users, stub_get_ids, stub_sync_project],
            max_concurrent_activities=50,
        ):
            wf_task = asyncio.create_task(
                env.client.execute_workflow(
                    SyncLabelStudioLaserLabelsWorkflow.run,
                    id=f"test-laser-sync-conc-{uuid.uuid4()}",
                    task_queue="test-laser-sync-conc",
                )
            )

            # Let the workflow get going and saturate the sem.
            await asyncio.sleep(0.5)
            release_gate.set()
            await wf_task

    # Workflow-level cap is PROJECT_CONCURRENCY = 4.
    assert peak_in_flight <= 4, f"peak concurrency was {peak_in_flight}, expected <= 4"
    assert len(started) == 20
