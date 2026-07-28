"""Activity: list every dive that needs model-assisted laser LS population.

Prediction-gated cohort (see the api's `needing-laser-population` endpoint):
HIGH-priority dives with an image that has a `LaserPrediction` and no
completed `LaserLabel`. Returns *all* matches so the scheduled populate
parent fans out one populate child per dive.
"""

from __future__ import annotations

from typing import List

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def select_dives_needing_laser_population_activity() -> List[int]:
    async with get_fs_client() as fs:
        dive_ids = await fs.dives.get_dives_needing_laser_population()

    activity.logger.info(
        "%d dive(s) need laser population: %s", len(dive_ids), dive_ids
    )
    return dive_ids
