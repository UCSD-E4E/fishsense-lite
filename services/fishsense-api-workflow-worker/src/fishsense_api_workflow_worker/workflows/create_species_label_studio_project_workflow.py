"""Workflow to idempotently create the species-labeling LS project."""

from datetime import timedelta

from temporalio import workflow


@workflow.defn
class CreateSpeciesLabelStudioProjectWorkflow:
    # pylint: disable=too-few-public-methods
    """Workflow to idempotently create the species-labeling LS project.

    Returns the project ID. Population of tasks is a separate
    `PopulateSpeciesLabelStudioProjectWorkflow`.
    """

    @workflow.run
    async def run(self) -> int:
        return await workflow.execute_activity(
            "create_species_label_studio_project_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
        )
