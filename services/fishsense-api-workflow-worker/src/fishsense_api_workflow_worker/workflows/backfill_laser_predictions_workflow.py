"""On-demand workflow: attach laser pre-annotations to existing laser Label
Studio tasks for one dive.

The predict parent calls the backfill activity after every persist, so a dive
it has just predicted (or re-predicted) gets its pre-annotations without any
operator action. This standalone workflow is for catching up a dive whose
predictions are in the database but never reached Label Studio — populate
seeds pre-annotations only at import time and will not re-run for an
already-populated dive, so every dive predicted before the backfill shipped is
in that state.

That is not a hypothetical: when this landed, prod held 1,555 predictions
across 31 dives, 183 of them outside the expected-laser region the stage now
enforces. Run it per dive_id:

    temporal workflow start \\
        --task-queue fishsense_api_queue \\
        --type BackfillLaserPredictionsWorkflow \\
        --workflow-id backfill-laser-predictions-<dive_id> \\
        --input <dive_id>

Idempotent: tasks already carrying a prediction from the current stage version
are skipped, so re-running costs a couple of Label Studio list calls.
"""

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.workflows._retry_policies import (
        SDK_FAIL_FAST_RETRY_POLICY,
    )


@workflow.defn
class BackfillLaserPredictionsWorkflow:
    # pylint: disable=too-few-public-methods
    """Attach a dive's persisted laser predictions to its existing LS tasks.

    Returns the number newly attached (0 when nothing was eligible or
    everything already carries the current version).
    """

    @workflow.run
    async def run(self, dive_id: int) -> int:
        return await workflow.execute_activity(
            "backfill_laser_predictions_for_dive_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=15),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
        )
