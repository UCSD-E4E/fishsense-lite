"""Activity to pick the next HIGH-priority dive that needs stage 2
species preprocessing.

Cohort: HIGH priority + has at least one PREDICTION cluster (so stage
1 has run) + has at least one image carrying a *valid* LaserLabel
(completed=True, superseded=False, both x/y populated) whose image
has no non-sentinel SpeciesLabel row. Cascades from valid lasers
like the headtail pipeline does (flipped 2026-05-05) so species
labeling fires in parallel with head/tail.

The selector is a single SDK call; the SQL predicate lives in the
api's `select-next/species-preprocessing` endpoint.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.cohort_selection import (
    select_next_dive,
)


@activity.defn
async def select_next_high_priority_dive_for_species_preprocessing_activity() -> int | None:
    return await select_next_dive(
        "species preprocessing",
        lambda fs: fs.dives.select_next_for_species_preprocessing(),
    )
