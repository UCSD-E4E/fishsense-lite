"""Workflow to populate every active laser-labeling LS project for one dive."""

import asyncio
from datetime import timedelta

from temporalio import workflow

from fishsense_shared import ExceptionGroupErrorLogging

PROJECT_CONCURRENCY = 4


@workflow.defn
class PopulateLaserLabelStudioProjectWorkflow:
    # pylint: disable=too-few-public-methods
    """Workflow to populate every active laser-labeling LS project for one dive.

    Active = "currently has at least one incomplete laser label" — so
    legacy/retired projects are excluded. The set is computed at run
    time rather than carried in config; project creation happens in a
    separate workflow.
    """

    @workflow.run
    async def run(self, dive_id: int) -> int:
        """Push laser tasks for `dive_id` into every active LS project.

        Returns the total number of tasks imported across all projects.
        """
        project_ids = await workflow.execute_activity(
            "get_active_laser_label_studio_project_ids_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
        )

        if not project_ids:
            workflow.logger.info(
                "No active laser LS projects found; nothing to populate"
            )
            return 0

        sem = asyncio.Semaphore(PROJECT_CONCURRENCY)
        results: list[int] = []

        async def _populate(project_id: int) -> None:
            async with sem:
                count = await workflow.execute_activity(
                    "populate_laser_label_studio_project_activity",
                    args=(dive_id, project_id),
                    schedule_to_close_timeout=timedelta(minutes=30),
                    heartbeat_timeout=timedelta(minutes=2),
                )
                results.append(count)

        async with ExceptionGroupErrorLogging(workflow.logger):
            async with asyncio.TaskGroup() as tg:
                for project_id in project_ids:
                    tg.create_task(_populate(project_id))

        return sum(results)
