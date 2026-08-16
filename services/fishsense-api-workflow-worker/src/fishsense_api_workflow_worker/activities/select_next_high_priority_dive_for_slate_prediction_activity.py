"""Activity to pick the next HIGH-priority dive needing slate predictions.

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


Cohort: HIGH-priority + `dive_slate_id` set + at least one slate frame
(`SpeciesLabel.content_of_image='Slate, Laser on slate'`) with no
`SlatePrediction` and no *completed* `DiveSlateLabel` (see
`select_next_for_slate_prediction` in the api's dive_controller). The CPU
slate-detector stage seeds a prediction per such frame; the dive-slate populate
step then serves it as a Label Studio pre-annotation (assisted review).

Single SDK call — the cohort answer is one query on the api side.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.cohort_selection import (
    select_next_dive,
)


@activity.defn
async def select_next_high_priority_dive_for_slate_prediction_activity() -> int | None:
    return await select_next_dive(
        "slate predictions",
        lambda fs: fs.dives.select_next_for_slate_prediction(),
    )
