"""On-demand workflow: attach head/tail pre-annotations to existing head/tail
Label Studio tasks for one dive.

The predict parent calls the backfill activity after every persist, so a dive
it has just predicted gets its pre-annotations without operator action. This
standalone workflow is for catching up a dive whose predictions are in the
database but never reached Label Studio — populate seeds pre-annotations only
at import time and will not re-run for an already-populated dive.

That is the normal case here, not an edge case. When this landed the corpus
held **39,766 existing head/tail tasks, 3,147 of them still unlabelled across
19 dives** — every one of those would keep a blank canvas no matter how good
the detector got, because their tasks were imported long before any prediction
existed. Run it per dive_id:

    temporal workflow start \\
        --task-queue fishsense_api_queue \\
        --type BackfillHeadtailPredictionsWorkflow \\
        --workflow-id backfill-headtail-predictions-<dive_id> \\
        --input <dive_id>

Idempotent: tasks already carrying a prediction from the current stage version
are skipped, so re-running costs one Label Studio list call per project.
"""

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.workflows._retry_policies import (
        SDK_FAIL_FAST_RETRY_POLICY,
    )


@workflow.defn
class BackfillHeadtailPredictionsWorkflow:
    # pylint: disable=too-few-public-methods
    """Attach a dive's persisted head/tail predictions to its existing LS tasks.

    Returns the number newly attached (0 when nothing was eligible or
    everything already carries the current version).
    """

    @workflow.run
    async def run(self, dive_id: int) -> int:
        return await workflow.execute_activity(
            "backfill_headtail_predictions_for_dive_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=15),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
        )
