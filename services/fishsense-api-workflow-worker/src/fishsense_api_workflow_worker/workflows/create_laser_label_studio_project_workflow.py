"""Workflow to idempotently create a per-dive laser-labeling LS project."""

from datetime import timedelta

from temporalio import workflow


@workflow.defn
class CreateLaserLabelStudioProjectWorkflow:
    # pylint: disable=too-few-public-methods
    """Workflow to idempotently create a per-dive laser-labeling LS project.

    Returns the project ID. Population of tasks is a separate
    `PopulateLaserLabelStudioProjectWorkflow`, which also calls the
    same create activity itself — manual invocation is rarely needed.
    """

    @workflow.run
    async def run(self, dive_id: int) -> int:
        return await workflow.execute_activity(
            "create_laser_label_studio_project_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
        )
