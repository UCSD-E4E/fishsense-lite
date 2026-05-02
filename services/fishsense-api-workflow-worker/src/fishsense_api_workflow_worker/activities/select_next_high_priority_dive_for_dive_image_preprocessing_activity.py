"""Activity to pick the next HIGH-priority dive that needs stage 2
dive-image preprocessing.

Cohort: HIGH priority + has at least one PREDICTION cluster (so stage
1 has run) + has at least one image without a completed species
label (matches `populate_species_label_studio_project_activity`'s
predicate, so populate consumes exactly what preprocess produces).

The selector is a single SDK call; the SQL predicate lives in the
api's `select-next/dive-image-preprocessing` endpoint.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def select_next_high_priority_dive_for_dive_image_preprocessing_activity() -> (
    int | None
):
    async with get_fs_client() as fs:
        dive_id = await fs.dives.select_next_for_dive_image_preprocessing()

    if dive_id is None:
        activity.logger.info(
            "no HIGH-priority dives needing dive-image preprocessing"
        )
    else:
        activity.logger.info(
            "next HIGH-priority dive needing dive-image preprocessing: dive_id=%d",
            dive_id,
        )
    return dive_id
