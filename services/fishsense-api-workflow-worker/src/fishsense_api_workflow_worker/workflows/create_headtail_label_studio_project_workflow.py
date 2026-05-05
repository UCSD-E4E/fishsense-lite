"""Workflow to idempotently create a per-dive headtail-labeling LS project."""

from datetime import timedelta

from temporalio import workflow


@workflow.defn
class CreateHeadTailLabelStudioProjectWorkflow:
    # pylint: disable=too-few-public-methods
    """Workflow to idempotently create a per-dive headtail-labeling LS project.

    Returns the project ID. Population of tasks is a separate
    `PopulateHeadTailLabelStudioProjectWorkflow`, which also calls the
    same create activity itself — manual invocation is rarely needed.
    """

    @workflow.run
    async def run(self, dive_id: int) -> int:
        return await workflow.execute_activity(
            "create_headtail_label_studio_project_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
        )
