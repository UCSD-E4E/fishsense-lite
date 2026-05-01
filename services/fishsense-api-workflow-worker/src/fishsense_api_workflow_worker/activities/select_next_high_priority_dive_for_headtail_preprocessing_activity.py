"""Activity to pick the next HIGH-priority dive that needs stage 5.1
head/tail preprocessing.

Cohort: HIGH priority + has at least one species label with
`top_three_photos_of_group=True` whose head/tail label is missing or
not yet completed (matches
`populate_headtail_label_studio_project_activity`'s predicate).

Returns the lowest dive_id in the cohort, or None. Ordering by `id`
is FIFO-ish; if dives ever get backfilled out of order, swap the
sort key to `dive_datetime`.
"""

from __future__ import annotations

from fishsense_api_sdk.models.priority import Priority
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def select_next_high_priority_dive_for_headtail_preprocessing_activity() -> (
    int | None
):
    async with get_fs_client() as fs:
        dives = await fs.dives.get() or []
        candidates = [
            d
            for d in dives
            if d.priority == Priority.HIGH and d.id is not None
        ]
        candidates.sort(key=lambda d: d.id)

        for dive in candidates:
            species_labels = await fs.labels.get_species_labels(dive.id) or []
            top_three_image_ids = {
                label.image_id
                for label in species_labels
                if label.top_three_photos_of_group
            }
            if not top_three_image_ids:
                continue

            existing_headtail = await fs.labels.get_headtail_labels(dive.id) or []
            completed_ids = {
                label.image_id for label in existing_headtail if label.completed
            }
            if top_three_image_ids - completed_ids:
                activity.logger.info(
                    "next HIGH-priority dive needing headtail preprocessing: dive_id=%d",
                    dive.id,
                )
                return dive.id

        activity.logger.info(
            "no HIGH-priority dives needing headtail preprocessing"
        )
        return None
