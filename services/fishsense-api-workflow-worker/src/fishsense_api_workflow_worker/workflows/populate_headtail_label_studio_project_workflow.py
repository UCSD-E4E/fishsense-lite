"""Workflow to populate the per-dive headtail-labeling LS project."""

from datetime import timedelta

from temporalio import workflow


@workflow.defn
class PopulateHeadTailLabelStudioProjectWorkflow:
    # pylint: disable=too-few-public-methods
    """Populate the per-dive headtail-labeling LS project for `dive_id`.

    Creates the per-dive project (idempotent title-match-or-create
    against `"{dive.name} - HeadTail Labeling"`) and pushes one LS
    task per still-unlabeled image in the dive.
    """

    @workflow.run
    async def run(self, dive_id: int) -> int:
        """Push headtail tasks for `dive_id` into the per-dive LS project.

        Returns the number of tasks imported.
        """
        project_id = await workflow.execute_activity(
            "create_headtail_label_studio_project_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
        )
        return await workflow.execute_activity(
            "populate_headtail_label_studio_project_activity",
            args=(dive_id, project_id),
            schedule_to_close_timeout=timedelta(minutes=30),
            heartbeat_timeout=timedelta(minutes=2),
        )
