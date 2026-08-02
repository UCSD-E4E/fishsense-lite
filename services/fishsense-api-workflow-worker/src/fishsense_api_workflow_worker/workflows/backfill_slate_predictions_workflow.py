"""On-demand workflow: backfill slate-detector pre-annotations onto existing
dive-slate Label Studio tasks for one dive.

The predict parent calls the backfill activity after every persist, so newly-
predicted dives get their pre-annotations automatically. This standalone
workflow is for catching up dives predicted *before* the backfill shipped (their
`SlatePrediction` rows are in the DB but never reached LS, because the populate
seeds pre-annotations only at import time and won't re-run for an already-
populated dive). Run it per dive_id:

    temporal workflow start \\
        --task-queue fishsense_api_queue \\
        --type BackfillSlatePredictionsWorkflow \\
        --workflow-id backfill-slate-predictions-<dive_id> \\
        --input <dive_id>
"""

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.workflows._retry_policies import (
        SDK_FAIL_FAST_RETRY_POLICY,
    )


@workflow.defn
class BackfillSlatePredictionsWorkflow:
    # pylint: disable=too-few-public-methods
    """Attach a dive's persisted slate predictions to its existing LS tasks.

    Returns the number of predictions newly attached (0 when nothing was
    eligible or everything was already attached).
    """

    @workflow.run
    async def run(self, dive_id: int) -> int:
        return await workflow.execute_activity(
            "backfill_slate_predictions_for_dive_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=15),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
        )
