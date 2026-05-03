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

    Target set is the union of:
      * The canonical project ID returned by
        `create_laser_label_studio_project_activity` (idempotent
        title-lookup-or-create, see `LASER_PROJECT_TITLE`).
      * Every project ID returned by the discovery query (LS projects
        currently holding at least one incomplete laser label).

    Bootstrap: a freshly-created project has zero LaserLabel rows, so
    the discovery query alone wouldn't pick it up — calling Create
    first and unioning its return solves the chicken-and-egg.
    Steady-state: the discovery query continues to cover legacy/
    additional projects (e.g. the prod laser project from before the
    Create XML was first checked in).
    """

    @workflow.run
    async def run(self, dive_id: int) -> int:
        """Push laser tasks for `dive_id` into every active LS project.

        Returns the total number of tasks imported across all projects.
        """
        canonical_project_id = await workflow.execute_activity(
            "create_laser_label_studio_project_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
        )
        discovered = await workflow.execute_activity(
            "get_active_laser_label_studio_project_ids_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
        )

        project_ids = sorted({canonical_project_id, *discovered})

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
