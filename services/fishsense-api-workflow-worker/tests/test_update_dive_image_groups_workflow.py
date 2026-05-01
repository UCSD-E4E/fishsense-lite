"""Workflow contract test for UpdateDiveImageGroupsWorkflow.

In-process Temporal worker with the activity stubbed; pins down the
single-activity dispatch contract and that `dive_id` is forwarded
through unchanged.
"""

from __future__ import annotations

import uuid

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.activities.update_dive_image_groups_activity import (  # noqa: E501  pylint: disable=line-too-long
    UpdateDiveImageGroupsResult,
)
from fishsense_api_workflow_worker.workflows.update_dive_image_groups_workflow import (  # noqa: E501  pylint: disable=line-too-long
    UpdateDiveImageGroupsWorkflow,
)


@pytest.mark.asyncio
async def test_workflow_forwards_dive_id_to_activity_and_returns_its_result():
    seen_args: list[int] = []

    @activity.defn(name="update_dive_image_groups_activity")
    async def stub_activity(dive_id: int) -> UpdateDiveImageGroupsResult:
        seen_args.append(dive_id)
        return UpdateDiveImageGroupsResult(
            skipped_already_grouped=False,
            new_clusters_created=3,
            species_labels_seen=12,
        )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-update-image-groups",
            workflows=[UpdateDiveImageGroupsWorkflow],
            activities=[stub_activity],
        ):
            result = await env.client.execute_workflow(
                UpdateDiveImageGroupsWorkflow.run,
                42,
                id=f"test-update-image-groups-{uuid.uuid4()}",
                task_queue="test-update-image-groups",
            )

    assert seen_args == [42]
    assert result.new_clusters_created == 3
    assert result.species_labels_seen == 12
    assert result.skipped_already_grouped is False
