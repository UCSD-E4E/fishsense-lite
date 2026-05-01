"""Workflow to idempotently create the dive-slate-labeling LS project."""

from datetime import timedelta

from temporalio import workflow


@workflow.defn
class CreateDiveSlateLabelStudioProjectWorkflow:
    # pylint: disable=too-few-public-methods
    """Workflow to idempotently create the slate-labeling LS project.

    Returns the project ID. Population of tasks is a separate
    `PopulateDiveSlateLabelStudioProjectWorkflow`.
    """

    @workflow.run
    async def run(self) -> int:
        return await workflow.execute_activity(
            "create_dive_slate_label_studio_project_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
        )
