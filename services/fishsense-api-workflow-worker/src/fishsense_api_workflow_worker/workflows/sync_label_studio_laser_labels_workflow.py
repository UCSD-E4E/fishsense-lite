"""Workflow to sync laser labels from Label Studio."""

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError

from fishsense_shared import ExceptionGroupErrorLogging

PROJECT_CONCURRENCY = 4

# Bound on concurrent per-dive validation child workflows. Each child
# is one SDK fetch + numpy line fit (~100 points), so the cap is sized
# to avoid pegging the api-worker on SDK roundtrips rather than CPU.
VALIDATION_CONCURRENCY = 8

DATA_PROCESSING_TASK_QUEUE = "fishsense_data_processing_queue"


@workflow.defn
class SyncLabelStudioLaserLabelsWorkflow:
    # pylint: disable=too-few-public-methods
    """Workflow to sync laser labels from Label Studio."""

    @workflow.run
    async def run(self):
        """Run the workflow to sync laser labels from Label Studio."""
        # pylint: disable=duplicate-code
        await workflow.execute_activity(
            "sync_users_label_studio_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=10),
        )

        label_studio_project_ids = await workflow.execute_activity(
            "get_laser_label_studio_project_ids_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=10),
        )

        sem = asyncio.Semaphore(PROJECT_CONCURRENCY)

        async def __run_for_project(project_id: int):
            async with sem:
                await workflow.execute_activity(
                    "sync_laser_labels_for_label_studio_project_activity",
                    args=(project_id,),
                    # Sized for the *first* run on a backlog project — the
                    # cursor is None initially, so we have to page every LS
                    # task even if the project is dormant. Once the cursor
                    # advances on a successful run, subsequent runs return
                    # almost instantly. ~7k LS pages fit in 2h at ~1s/page.
                    schedule_to_close_timeout=timedelta(hours=2),
                    heartbeat_timeout=timedelta(minutes=2),
                )

        async with ExceptionGroupErrorLogging(workflow.logger):
            async with asyncio.TaskGroup() as tg:
                for project_id in label_studio_project_ids:
                    tg.create_task(__run_for_project(project_id))

        # Post-sync validation pass. Once the per-project syncs land,
        # walk dives whose laser labeling is complete and fan out a
        # RANSAC line-fit child workflow per dive on the data-worker.
        # Phase 1 is observe-only — the child logs outlier labels but
        # doesn't write `superseded`. Failures inside the validation
        # pass shouldn't roll back a successful sync, so the pass is
        # wrapped in its own ExceptionGroup logger.
        complete_dive_ids = await workflow.execute_activity(
            "get_dives_with_complete_laser_labeling_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=10),
        )

        validation_sem = asyncio.Semaphore(VALIDATION_CONCURRENCY)

        async def __validate_dive(dive_id: int):
            async with validation_sem:
                try:
                    await workflow.execute_child_workflow(
                        "ValidateLaserLabelsForDiveWorkflow",
                        dive_id,
                        # Re-firing on the same complete dive every hour
                        # is wasted work but cheap (one SDK fetch +
                        # numpy fit). ALLOW_DUPLICATE so re-runs aren't
                        # blocked by the prior firing's success.
                        id=f"validate-laser-labels-{dive_id}",
                        task_queue=DATA_PROCESSING_TASK_QUEUE,
                        # Must be ≥ the child activity's
                        # schedule_to_close (15m); see
                        # validate_laser_labels_for_dive_workflow.py.
                        execution_timeout=timedelta(minutes=20),
                        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
                    )
                except WorkflowAlreadyStartedError:
                    workflow.logger.info(
                        "validate-laser-labels-%d already running; "
                        "skipping duplicate dispatch",
                        dive_id,
                    )

        # suppress=True: validation failures must not roll back a successful
        # sync. Without it, the BaseExceptionGroup from asyncio.TaskGroup
        # leaks out of the workflow as a non-FailureError, which Temporal
        # classifies as WORKFLOW_TASK_FAILED_CAUSE_WORKFLOW_WORKER_UNHANDLED_FAILURE
        # and retries forever.
        async with ExceptionGroupErrorLogging(workflow.logger, suppress=True):
            async with asyncio.TaskGroup() as tg:
                for dive_id in complete_dive_ids:
                    tg.create_task(__validate_dive(dive_id))
