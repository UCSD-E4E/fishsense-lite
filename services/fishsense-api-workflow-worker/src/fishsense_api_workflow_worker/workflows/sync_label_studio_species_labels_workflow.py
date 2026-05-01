"""Workflow to sync species labels from Label Studio (stage 4.2)."""

import asyncio
from datetime import timedelta

from temporalio import workflow

from fishsense_shared import ExceptionGroupErrorLogging

PROJECT_CONCURRENCY = 4


@workflow.defn
class SyncLabelStudioSpeciesLabelsWorkflow:
    # pylint: disable=too-few-public-methods
    """Workflow to sync species labels from Label Studio."""

    @workflow.run
    async def run(self):
        """Run the workflow to sync species labels from Label Studio."""
        # pylint: disable=duplicate-code
        await workflow.execute_activity(
            "sync_users_label_studio_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=10),
        )

        label_studio_project_ids = await workflow.execute_activity(
            "get_species_label_studio_project_ids_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=10),
        )

        sem = asyncio.Semaphore(PROJECT_CONCURRENCY)

        async def __run_for_project(project_id: int):
            async with sem:
                await workflow.execute_activity(
                    "sync_species_labels_for_label_studio_project_activity",
                    args=(project_id,),
                    # Sized for first-run-on-backlog: per-project sync pages
                    # the full LS task list before the cursor is set.
                    # Species projects can hold thousands of tasks per dive
                    # cohort; keep parity with laser/headtail at 2h.
                    schedule_to_close_timeout=timedelta(hours=2),
                    heartbeat_timeout=timedelta(minutes=2),
                )

        async with ExceptionGroupErrorLogging(workflow.logger):
            async with asyncio.TaskGroup() as tg:
                for project_id in label_studio_project_ids:
                    tg.create_task(__run_for_project(project_id))
