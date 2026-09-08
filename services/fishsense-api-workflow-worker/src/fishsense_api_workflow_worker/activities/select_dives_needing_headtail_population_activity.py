"""Activity to list the dives needing model-assisted head/tail population.

Prediction-gated cohort: HIGH priority + at least one canonical image with a
valid laser dot, a `HeadTailPrediction`, and no completed `HeadTailLabel`. See
`select_dives_needing_headtail_population` in the api's
`dive_prediction_cohort_controller`.

Returns every match — the scheduled parent fans out one populate child per
dive, rather than draining one dive per firing like the predict parents do.
"""

from __future__ import annotations

from typing import List

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def select_dives_needing_headtail_population_activity() -> List[int]:
    async with get_fs_client() as fs:
        dive_ids = await fs.dives.get_dives_needing_headtail_population()

    activity.logger.info(
        "%d dive(s) need headtail population: %s", len(dive_ids), dive_ids
    )
    return dive_ids
