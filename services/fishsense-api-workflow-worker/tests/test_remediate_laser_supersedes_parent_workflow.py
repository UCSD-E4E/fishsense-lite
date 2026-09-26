"""Workflow contract for `RemediateLaserSupersedesParentWorkflow`.

The light worker scales to zero, so a child dispatched without waking it
first does not fail — it hangs until its execution timeout. The parent must
wake it, then run the child on the REAL light queue (the stub polls the shared
constant, not a literal), and hand the report back untouched.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from fishsense_shared import DATA_PROCESSING_LIGHT_TASK_QUEUE
from fishsense_shared.laser_remediation import (
    CHILD_WORKFLOW,
    RemediateLaserSupersedesInput,
)
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows.remediate_laser_supersedes_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    RemediateLaserSupersedesParentWorkflow,
)

pytestmark = pytest.mark.asyncio

EVENTS: list = []


@workflow.defn(name=CHILD_WORKFLOW)
class _StubChild:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, request: RemediateLaserSupersedesInput) -> dict:
        return await workflow.execute_activity(
            "stub_child_body",
            args=(request,),
            schedule_to_close_timeout=timedelta(minutes=1),
        )


async def test_wakes_the_light_worker_then_returns_the_childs_report():
    EVENTS.clear()

    @activity.defn(name="ensure_light_worker_running_activity")
    async def wake() -> int:
        EVENTS.append("wake")
        return 1

    @activity.defn(name="stub_child_body")
    async def child_body(request: RemediateLaserSupersedesInput) -> dict:
        EVENTS.append(("child", request.dive_ids, request.apply))
        return {"mode": "dry_run", "plan_sha256": "abc"}

    request = RemediateLaserSupersedesInput(dive_ids=[7, 9], excluded_dive_ids=[77])
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-remediate-parent",
            workflows=[RemediateLaserSupersedesParentWorkflow],
            activities=[wake],
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_LIGHT_TASK_QUEUE,
            workflows=[_StubChild],
            activities=[child_body],
        ):
            report = await env.client.execute_workflow(
                RemediateLaserSupersedesParentWorkflow.run,
                request,
                id=f"test-remediate-parent-{uuid.uuid4()}",
                task_queue="test-remediate-parent",
            )

    assert EVENTS == ["wake", ("child", [7, 9], False)]
    assert report == {"mode": "dry_run", "plan_sha256": "abc"}
