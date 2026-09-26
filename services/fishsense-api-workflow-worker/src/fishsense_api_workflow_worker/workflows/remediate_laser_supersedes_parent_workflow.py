"""Operator entry point for the laser-supersede remediation run.

On-demand, never scheduled. Wakes the light worker (it scales to zero, and a
child dispatched onto an unserved queue hangs rather than fails), runs the
data-worker's `RemediateLaserSupersedesWorkflow` there, and returns its report.
Dry run unless the input carries `apply` and the reviewed plan's digest — see
that workflow. Start it through `tools/remediate_laser_supersedes.py`.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from fishsense_shared import DATA_PROCESSING_LIGHT_TASK_QUEUE
    from fishsense_shared.laser_remediation import (
        CHILD_WORKFLOW,
        PARENT_WORKFLOW,
        RemediateLaserSupersedesInput,
    )


@workflow.defn(name=PARENT_WORKFLOW)
class RemediateLaserSupersedesParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Wake the light worker, then plan (or apply) on it."""

    @workflow.run
    async def run(self, request: RemediateLaserSupersedesInput) -> dict:
        await workflow.execute_activity(
            "ensure_light_worker_running_activity",
            schedule_to_close_timeout=timedelta(minutes=5),
        )
        return await workflow.execute_child_workflow(
            CHILD_WORKFLOW,
            request,
            id=f"remediate-laser-supersedes-{workflow.info().workflow_id}",
            task_queue=DATA_PROCESSING_LIGHT_TASK_QUEUE,
            # ~270 dives, 8 at a time, one API read each: minutes, not hours.
            execution_timeout=timedelta(hours=2),
        )
