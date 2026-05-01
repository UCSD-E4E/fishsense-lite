"""Activity to pick the next HIGH-priority dive that needs stage 2
dive-image preprocessing.

Cohort: HIGH priority + has at least one PREDICTION cluster (so stage
1 has run) + has at least one image without a completed species
label (matches `populate_species_label_studio_project_activity`'s
predicate, so populate consumes exactly what preprocess produces).

Lives on the api-worker so the SDK call runs on the orchestrator's
docker network. Returns the lowest dive_id in the cohort, or None
when the cohort is empty. Ordering by `id` is FIFO-ish; if dives
ever get backfilled out of order, swap the sort key to
`dive_datetime`.
"""

from __future__ import annotations

from fishsense_api_sdk.models.data_source import DataSource
from fishsense_api_sdk.models.priority import Priority
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def select_next_high_priority_dive_for_dive_image_preprocessing_activity() -> (
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
            clusters = (
                await fs.images.get_clusters(dive.id, DataSource.PREDICTION.value)
                or []
            )
            if not clusters:
                continue

            images = await fs.images.get(dive_id=dive.id) or []
            existing_labels = await fs.labels.get_species_labels(dive.id) or []
            completed_ids = {
                label.image_id for label in existing_labels if label.completed
            }
            has_incomplete = any(
                image.id not in completed_ids for image in images
            )
            if has_incomplete:
                activity.logger.info(
                    "next HIGH-priority dive needing dive-image preprocessing: dive_id=%d",
                    dive.id,
                )
                return dive.id

        activity.logger.info(
            "no HIGH-priority dives needing dive-image preprocessing"
        )
        return None
