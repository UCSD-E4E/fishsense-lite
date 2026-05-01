"""Activity to pick the next HIGH-priority dive that needs stage 9
slate preprocessing.

Cohort: HIGH priority + has a `dive_slate_id` set + has at least one
species label with `content_of_image == 'Slate, Laser on slate'` whose
dive_slate_label is missing or not yet completed (matches
`populate_dive_slate_label_studio_project_activity`'s predicate).

Returns the lowest dive_id in the cohort, or None. Ordering by `id`
is FIFO-ish; if dives ever get backfilled out of order, swap the
sort key to `dive_datetime`.
"""

from __future__ import annotations

from fishsense_api_sdk.models.priority import Priority
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client

SLATE_CONTENT_MARKER = "Slate, Laser on slate"


@activity.defn
async def select_next_high_priority_dive_for_slate_preprocessing_activity() -> (
    int | None
):
    async with get_fs_client() as fs:
        dives = await fs.dives.get() or []
        candidates = [
            d
            for d in dives
            if d.priority == Priority.HIGH
            and d.id is not None
            and d.dive_slate_id is not None
        ]
        candidates.sort(key=lambda d: d.id)

        for dive in candidates:
            species_labels = await fs.labels.get_species_labels(dive.id) or []
            slate_marked_image_ids = {
                label.image_id
                for label in species_labels
                if label.content_of_image == SLATE_CONTENT_MARKER
            }
            if not slate_marked_image_ids:
                continue

            existing_slate = await fs.labels.get_dive_slate_labels(dive.id) or []
            completed_ids = {
                label.image_id for label in existing_slate if label.completed
            }
            if slate_marked_image_ids - completed_ids:
                activity.logger.info(
                    "next HIGH-priority dive needing slate preprocessing: dive_id=%d",
                    dive.id,
                )
                return dive.id

        activity.logger.info(
            "no HIGH-priority dives needing slate preprocessing"
        )
        return None
