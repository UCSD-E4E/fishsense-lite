"""Workflow to populate every active species-labeling LS project for one dive."""

import asyncio
from datetime import timedelta

from temporalio import workflow

from fishsense_shared import ExceptionGroupErrorLogging

PROJECT_CONCURRENCY = 4


@workflow.defn
class PopulateSpeciesLabelStudioProjectWorkflow:
    # pylint: disable=too-few-public-methods
    """Workflow to populate every active species-labeling LS project for one dive.

    Active = "currently has at least one incomplete species label."
    Project creation is a separate workflow's responsibility.
    """

    @workflow.run
    async def run(self, dive_id: int) -> int:
        """Push species tasks for `dive_id` into every active LS project.

        Returns the total number of tasks imported across all projects.
        """
        project_ids = await workflow.execute_activity(
            "get_active_species_label_studio_project_ids_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
        )

        if not project_ids:
            workflow.logger.info(
                "No active species LS projects found; nothing to populate"
            )
            return 0

        sem = asyncio.Semaphore(PROJECT_CONCURRENCY)
        results: list[int] = []

        async def _populate(project_id: int) -> None:
            async with sem:
                count = await workflow.execute_activity(
                    "populate_species_label_studio_project_activity",
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
