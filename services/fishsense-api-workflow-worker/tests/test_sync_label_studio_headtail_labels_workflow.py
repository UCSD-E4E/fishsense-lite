"""Workflow contract test for SyncLabelStudioHeadTailLabelsWorkflow.

Mirror of the laser-sync contract test; lighter on assertions because
the two workflows are structurally identical (Phase 3 will fold them).
"""

from __future__ import annotations

import uuid
from typing import List

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows.sync_label_studio_headtail_labels_workflow import (
    SyncLabelStudioHeadTailLabelsWorkflow,
)


@pytest.mark.asyncio
async def test_workflow_invokes_one_sync_per_project():
    sync_calls: List[int] = []

    @activity.defn(name="sync_users_label_studio_activity")
    async def stub_sync_users() -> None:
        return None

    @activity.defn(name="get_headtail_label_studio_project_ids_activity")
    async def stub_get_ids() -> List[int]:
        return [11, 22]

    @activity.defn(name="sync_headtail_labels_for_label_studio_project_activity")
    async def stub_sync_project(project_id: int) -> None:
        sync_calls.append(project_id)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-headtail-sync",
            workflows=[SyncLabelStudioHeadTailLabelsWorkflow],
            activities=[stub_sync_users, stub_get_ids, stub_sync_project],
        ):
            await env.client.execute_workflow(
                SyncLabelStudioHeadTailLabelsWorkflow.run,
                id=f"test-headtail-sync-{uuid.uuid4()}",
                task_queue="test-headtail-sync",
            )

    assert sorted(sync_calls) == [11, 22]
