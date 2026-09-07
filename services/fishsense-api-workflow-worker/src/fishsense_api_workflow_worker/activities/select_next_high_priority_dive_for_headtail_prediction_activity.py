"""Activity to pick the next HIGH-priority dive needing head/tail predictions.

Cohort: HIGH-priority + at least one canonical image with a *valid* laser dot,
no live human `HeadTailLabel`, and either no `HeadTailPrediction` or a stale
one — a `predictor_version` mismatch, or a prediction made from a laser label
since superseded. See `select_next_for_headtail_prediction` in the api's
`dive_prediction_cohort_controller`.

Single SDK call — the cohort answer is one query on the api side.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.cohort_selection import (
    select_next_dive,
)


@activity.defn
async def select_next_high_priority_dive_for_headtail_prediction_activity() -> (
    int | None
):
    return await select_next_dive(
        "headtail predictions",
        lambda fs: fs.dives.select_next_for_headtail_prediction(),
    )
