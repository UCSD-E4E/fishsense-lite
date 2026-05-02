"""Activity to pick the next HIGH-priority dive that needs stage 13
laser calibration.

Cohort: HIGH priority + has `dive_slate_id` set + no `LaserExtrinsics`
row yet + at least `MIN_COMPLETED_SLATE_LABELS` completed
`DiveSlateLabel` rows. The minimum-2 floor matches the data-worker
activity's `MIN_LASER_POINTS = 2` precondition; selecting a dive with
fewer than two completed slate labels would dispatch a child that
raises `ValueError` and re-fires every hour, since `put_laser_extrinsics`
never gets written.

Returns the lowest dive_id in the cohort, or None. Ordering by `id`
is FIFO-ish; if dives ever get backfilled out of order, swap the
sort key to `dive_datetime`.
"""

from __future__ import annotations

from fishsense_api_sdk.models.priority import Priority
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client

MIN_COMPLETED_SLATE_LABELS = 2


@activity.defn
async def select_next_high_priority_dive_for_laser_calibration_activity() -> (
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
            extrinsics = await fs.dives.get_laser_extrinsics(dive.id)
            if extrinsics is not None:
                continue

            slate_labels = await fs.labels.get_dive_slate_labels(dive.id) or []
            completed = [label for label in slate_labels if label.completed]
            if len(completed) >= MIN_COMPLETED_SLATE_LABELS:
                activity.logger.info(
                    "next HIGH-priority dive needing laser calibration: dive_id=%d",
                    dive.id,
                )
                return dive.id

        activity.logger.info(
            "no HIGH-priority dives needing laser calibration"
        )
        return None
