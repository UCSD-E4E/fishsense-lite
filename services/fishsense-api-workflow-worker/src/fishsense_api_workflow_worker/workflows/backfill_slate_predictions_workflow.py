"""On-demand workflow: backfill slate-detector pre-annotations onto existing
dive-slate Label Studio tasks for one dive.

**RETIRED 2026-08-03 — registered, but nothing schedules it.** The
ECC >= 0.80 acceptance gate does not transfer out of distribution: pool
dives produced high-ECC (0.93-0.97) *false* fits that sailed through it
(prod dives 65/71/77/80/83, all pool). The team declined an
active-learning loop; `predict-slate-images-workflow-schedule` is now
actively deleted at worker startup (`worker._RETIRED_SCHEDULE_IDS`) and
the 130 seeded Label Studio predictions were removed.

The code is kept registered so a future evaluation can start it by hand
— it is dormant, not dead — but nothing invokes it on its own. Do not
read it as part of the live pipeline.

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
