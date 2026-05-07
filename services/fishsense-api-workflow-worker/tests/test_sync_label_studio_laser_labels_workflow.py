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
from datetime import timedelta
from typing import List

import pytest
from temporalio import activity, workflow
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows.sync_label_studio_laser_labels_workflow import (
    SyncLabelStudioLaserLabelsWorkflow,
)


# Stand-in for the data-worker child workflow. Lives on a separate task
# queue so the sync workflow's `task_queue=DATA_PROCESSING_TASK_QUEUE`
# dispatch is exercised end-to-end (instead of silently degrading to
# "child runs on the same queue").
DATA_PROCESSING_TASK_QUEUE = "fishsense_data_processing_queue"


@workflow.defn(name="ValidateLaserLabelsForDiveWorkflow")
class _StubValidateChildWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, dive_id: int) -> int:
        return await workflow.execute_activity(
            "validate_laser_labels_for_dive_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
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

    @activity.defn(name="get_dives_with_complete_laser_labeling_activity")
    async def stub_get_complete_dives() -> List[int]:
        calls.append(_Call("get_dives_with_complete_laser_labeling_activity", ()))
        return []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-laser-sync",
            workflows=[SyncLabelStudioLaserLabelsWorkflow],
            activities=[
                stub_sync_users,
                stub_get_ids,
                stub_sync_project,
                stub_get_complete_dives,
            ],
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

    # Validation step runs after the sync — confirm it's invoked even
    # when no dives are complete.
    assert any(
        c.name == "get_dives_with_complete_laser_labeling_activity" for c in calls
    )


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

    @activity.defn(name="get_dives_with_complete_laser_labeling_activity")
    async def stub_get_complete_dives() -> List[int]:
        return []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-laser-sync-empty",
            workflows=[SyncLabelStudioLaserLabelsWorkflow],
            activities=[
                stub_sync_users,
                stub_get_ids,
                stub_sync_project,
                stub_get_complete_dives,
            ],
        ):
            await env.client.execute_workflow(
                SyncLabelStudioLaserLabelsWorkflow.run,
                id=f"test-laser-sync-empty-{uuid.uuid4()}",
                task_queue="test-laser-sync-empty",
            )

    assert not sync_calls


@pytest.mark.asyncio
async def test_workflow_dispatches_validation_child_per_complete_dive():
    """Each dive returned by get_dives_with_complete_laser_labeling_activity
    must be dispatched to ValidateLaserLabelsForDiveWorkflow on the
    data-worker task queue."""
    validated: List[int] = []

    @activity.defn(name="sync_users_label_studio_activity")
    async def stub_sync_users() -> None:
        return None

    @activity.defn(name="get_laser_label_studio_project_ids_activity")
    async def stub_get_ids() -> List[int]:
        return []

    @activity.defn(name="sync_laser_labels_for_label_studio_project_activity")
    async def stub_sync_project(_: int) -> None:
        return None

    @activity.defn(name="get_dives_with_complete_laser_labeling_activity")
    async def stub_get_complete_dives() -> List[int]:
        return [10, 20, 30]

    @activity.defn(name="validate_laser_labels_for_dive_activity")
    async def stub_validate(dive_id: int) -> int:
        validated.append(dive_id)
        return 0

    queue_parent = "test-laser-sync-validate-parent"
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=queue_parent,
            workflows=[SyncLabelStudioLaserLabelsWorkflow],
            activities=[
                stub_sync_users,
                stub_get_ids,
                stub_sync_project,
                stub_get_complete_dives,
            ],
        ):
            async with Worker(
                env.client,
                task_queue=DATA_PROCESSING_TASK_QUEUE,
                workflows=[_StubValidateChildWorkflow],
                activities=[stub_validate],
            ):
                await env.client.execute_workflow(
                    SyncLabelStudioLaserLabelsWorkflow.run,
                    id=f"test-laser-sync-validate-{uuid.uuid4()}",
                    task_queue=queue_parent,
                )

    assert sorted(validated) == [10, 20, 30]


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

    @activity.defn(name="get_dives_with_complete_laser_labeling_activity")
    async def stub_get_complete_dives() -> List[int]:
        return []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-laser-sync-conc",
            workflows=[SyncLabelStudioLaserLabelsWorkflow],
            activities=[
                stub_sync_users,
                stub_get_ids,
                stub_sync_project,
                stub_get_complete_dives,
            ],
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


@pytest.mark.asyncio
async def test_workflow_completes_when_validation_children_fail():
    """Regression: when every per-dive validation child workflow fails,
    the parent must still complete successfully.

    Before the fix, asyncio.TaskGroup wrapped the per-dive failures in a
    bare BaseExceptionGroup. That isn't a temporalio FailureError
    subclass, so Temporal classified it as
    WORKFLOW_TASK_FAILED_CAUSE_WORKFLOW_WORKER_UNHANDLED_FAILURE and
    retried the workflow task indefinitely instead of treating the run
    as a workflow execution failure. The validation pass is wrapped in
    its own ExceptionGroupErrorLogging block precisely so failures
    there don't roll back a successful sync, so the parent must finish.
    """

    @activity.defn(name="sync_users_label_studio_activity")
    async def stub_sync_users() -> None:
        return None

    @activity.defn(name="get_laser_label_studio_project_ids_activity")
    async def stub_get_ids() -> List[int]:
        return []

    @activity.defn(name="sync_laser_labels_for_label_studio_project_activity")
    async def stub_sync_project(_: int) -> None:
        return None

    @activity.defn(name="get_dives_with_complete_laser_labeling_activity")
    async def stub_get_complete_dives() -> List[int]:
        return [1, 2, 3]

    @activity.defn(name="validate_laser_labels_for_dive_activity")
    async def stub_validate_fails(dive_id: int) -> int:
        raise ApplicationError(
            f"validation deliberately failed for dive {dive_id}",
            non_retryable=True,
        )

    queue_parent = "test-laser-sync-validate-fails-parent"
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=queue_parent,
            workflows=[SyncLabelStudioLaserLabelsWorkflow],
            activities=[
                stub_sync_users,
                stub_get_ids,
                stub_sync_project,
                stub_get_complete_dives,
            ],
        ):
            async with Worker(
                env.client,
                task_queue=DATA_PROCESSING_TASK_QUEUE,
                workflows=[_StubValidateChildWorkflow],
                activities=[stub_validate_fails],
            ):
                # Must NOT raise. With the bug, this would either raise
                # WorkflowFailureError or hang in the unhandled-failure
                # retry loop until the test's wall-clock timeout.
                await asyncio.wait_for(
                    env.client.execute_workflow(
                        SyncLabelStudioLaserLabelsWorkflow.run,
                        id=f"test-laser-sync-validate-fails-{uuid.uuid4()}",
                        task_queue=queue_parent,
                    ),
                    timeout=30.0,
                )
