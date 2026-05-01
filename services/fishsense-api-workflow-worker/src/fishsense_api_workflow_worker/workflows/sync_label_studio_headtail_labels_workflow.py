"""Workflow to sync laser head tail from Label Studio."""

import asyncio
from datetime import timedelta

from temporalio import workflow

from fishsense_shared import ExceptionGroupErrorLogging

PROJECT_CONCURRENCY = 4


@workflow.defn
class SyncLabelStudioHeadTailLabelsWorkflow:
    # pylint: disable=too-few-public-methods
    """Workflow to sync laser head tail from Label Studio."""

    @workflow.run
    async def run(self):
        """Run the workflow to sync laser head tail from Label Studio."""
        # pylint: disable=duplicate-code
        await workflow.execute_activity(
            "sync_users_label_studio_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=10),
        )

        label_studio_project_ids = await workflow.execute_activity(
            "get_headtail_label_studio_project_ids_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=10),
        )

        sem = asyncio.Semaphore(PROJECT_CONCURRENCY)

        async def __run_for_project(project_id: int):
            async with sem:
                await workflow.execute_activity(
                    "sync_headtail_labels_for_label_studio_project_activity",
                    args=(project_id,),
                    schedule_to_close_timeout=timedelta(minutes=30),
                    heartbeat_timeout=timedelta(minutes=2),
                )

        async with ExceptionGroupErrorLogging(workflow.logger):
            async with asyncio.TaskGroup() as tg:
                for project_id in label_studio_project_ids:
                    tg.create_task(__run_for_project(project_id))
