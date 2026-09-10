"""Workflow to populate the per-dive laser-labeling LS project."""

from temporalio import workflow

from fishsense_api_workflow_worker.workflows._populate import create_then_populate


@workflow.defn
class PopulateLaserLabelStudioProjectWorkflow:
    # pylint: disable=too-few-public-methods
    """Populate the per-dive laser-labeling LS project for `dive_id`.

    Creates the per-dive project (idempotent title-match-or-create
    against `"{dive.name} - Laser Calibration Labeling"`) and pushes
    one LS task per still-unlabeled image in the dive.
    """

    @workflow.run
    async def run(self, dive_id: int) -> int:
        """Push laser tasks for `dive_id` into the per-dive LS project.

        Returns the number of tasks imported.
        """
        return await create_then_populate("laser", dive_id)
