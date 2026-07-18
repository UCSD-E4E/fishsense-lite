"""Activity: list every dive that needs species LS tasks (re)populated.

Superseded-aware cohort (see the api's `needing-species-population`
endpoint): HIGH-priority dives with a laser-valid image that has no
non-superseded, real-project SpeciesLabel row. Returns *all* matches so
the scheduled populate parent fans out one populate child per dive.
Unlike the species-preprocessing selector, a dive whose old-project
species rows were superseded (post hosted-LS migration) re-enters the
cohort here.
"""

from __future__ import annotations

from typing import List

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def select_dives_needing_species_population_activity() -> List[int]:
    async with get_fs_client() as fs:
        dive_ids = await fs.dives.get_dives_needing_species_population()

    activity.logger.info(
        "%d dive(s) need species population: %s",
        len(dive_ids),
        dive_ids,
    )
    return dive_ids
