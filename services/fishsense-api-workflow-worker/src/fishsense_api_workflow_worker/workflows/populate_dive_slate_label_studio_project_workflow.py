"""Workflow to populate every active dive-slate-labeling LS project for one dive."""

import asyncio
from datetime import timedelta

from temporalio import workflow

from fishsense_shared import ExceptionGroupErrorLogging

PROJECT_CONCURRENCY = 4


@workflow.defn
class PopulateDiveSlateLabelStudioProjectWorkflow:
    # pylint: disable=too-few-public-methods
    """Workflow to populate every active dive-slate-labeling LS project for one dive.

    Target set is the union of the canonical project (Create) and any
    additional projects currently holding incomplete dive-slate labels.
    See `PopulateLaserLabelStudioProjectWorkflow` for the bootstrap
    rationale.
    """

    @workflow.run
    async def run(self, dive_id: int) -> int:
        """Push slate tasks for `dive_id` into every active LS project.

        Returns the total number of tasks imported across all projects.
        """
        canonical_project_id = await workflow.execute_activity(
            "create_dive_slate_label_studio_project_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
        )
        discovered = await workflow.execute_activity(
            "get_active_dive_slate_label_studio_project_ids_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
        )

        project_ids = sorted({canonical_project_id, *discovered})

        sem = asyncio.Semaphore(PROJECT_CONCURRENCY)
        results: list[int] = []

        async def _populate(project_id: int) -> None:
            async with sem:
                count = await workflow.execute_activity(
                    "populate_dive_slate_label_studio_project_activity",
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
