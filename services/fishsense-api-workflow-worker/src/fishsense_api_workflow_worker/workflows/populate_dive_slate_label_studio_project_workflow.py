"""Workflow to populate the per-dive slate-labeling LS project."""

from temporalio import workflow

from fishsense_api_workflow_worker.workflows._populate import create_then_populate


@workflow.defn
class PopulateDiveSlateLabelStudioProjectWorkflow:
    # pylint: disable=too-few-public-methods
    """Populate the per-dive slate-labeling LS project for `dive_id`.

    Creates the per-dive project (idempotent title-match-or-create
    against `"{dive.name} - Dive Slate Labeling"`) and pushes one LS
    task per still-unlabeled slate image in the dive.
    """

    @workflow.run
    async def run(self, dive_id: int) -> int:
        """Push slate tasks for `dive_id` into the per-dive LS project.

        Returns the number of tasks imported.
        """
        return await create_then_populate("dive_slate", dive_id)
