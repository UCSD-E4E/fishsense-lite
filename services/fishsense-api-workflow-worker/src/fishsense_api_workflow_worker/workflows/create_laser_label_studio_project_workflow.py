"""Workflow to idempotently create the laser-labeling LS project."""

from datetime import timedelta

from temporalio import workflow


@workflow.defn
class CreateLaserLabelStudioProjectWorkflow:
    # pylint: disable=too-few-public-methods
    """Workflow to idempotently create the laser-labeling LS project.

    Returns the project ID. Run once during initial deployment, or
    any time you need to confirm the project exists; re-runs are safe.
    Population of tasks is a separate `PopulateLaserLabelStudioProjectWorkflow`.
    """

    @workflow.run
    async def run(self) -> int:
        return await workflow.execute_activity(
            "create_laser_label_studio_project_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
        )
