"""Workflow to sync laser labels from Label Studio."""

import asyncio
from datetime import timedelta

from temporalio import workflow

from fishsense_api_workflow_worker.exception_group_error_logging import (
    ExceptionGroupErrorLogging,
)


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

        with ExceptionGroupErrorLogging(workflow.logger):
            async with asyncio.TaskGroup() as tg:
                for project_id in label_studio_project_ids:
                    tg.create_task(
                        workflow.execute_activity(
                            "sync_laser_labels_for_label_studio_project_activity",
                            args=(project_id,),
                            schedule_to_close_timeout=timedelta(minutes=30),
                        )
                    )
