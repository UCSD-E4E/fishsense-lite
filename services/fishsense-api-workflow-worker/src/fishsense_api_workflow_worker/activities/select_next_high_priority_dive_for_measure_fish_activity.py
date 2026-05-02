"""Activity to pick the next HIGH-priority dive that needs stage 14
fish measurement.

Cohort: HIGH priority + has `LaserExtrinsics` (stage 13 done) + has at
least one `data_source=LABEL_STUDIO` cluster (stage 6.1 done) where
`fish_id` is None — i.e., a cluster that hasn't yet been bound to a
fish by `measure_fish_activity._ensure_fish`.

The `fish_id is None` predicate is a first-run gate: after a successful
measurement run, every reachable cluster has `fish_id` set, so the dive
falls out of the cohort. **It is not a strict idempotency gate** —
`measure_fish_activity` is non-idempotent (`post_measurement` is a POST,
not a PUT, and the SDK has no per-image measurement query), so a
partially-failed run that left some clusters bound and others not will
re-fire and *duplicate* measurements on the already-bound clusters.

Because of that, the parent workflow is registered but **deliberately
not scheduled** — operators trigger it on-demand per dive once the
upstream context is ready. This selector exists so on-demand callers
can ask "which dive is next?" without re-implementing the predicate.

Returns the lowest dive_id in the cohort, or None.
"""

from __future__ import annotations

from fishsense_api_sdk.models.data_source import DataSource
from fishsense_api_sdk.models.priority import Priority
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def select_next_high_priority_dive_for_measure_fish_activity() -> int | None:
    async with get_fs_client() as fs:
        dives = await fs.dives.get() or []
        candidates = [
            d
            for d in dives
            if d.priority == Priority.HIGH and d.id is not None
        ]
        candidates.sort(key=lambda d: d.id)

        for dive in candidates:
            extrinsics = await fs.dives.get_laser_extrinsics(dive.id)
            if extrinsics is None:
                continue

            clusters = (
                await fs.images.get_clusters(
                    dive.id, DataSource.LABEL_STUDIO.value
                )
                or []
            )
            unbound = [c for c in clusters if c.fish_id is None]
            if unbound:
                activity.logger.info(
                    "next HIGH-priority dive needing measurement: dive_id=%d "
                    "(unbound_clusters=%d)",
                    dive.id,
                    len(unbound),
                )
                return dive.id

        activity.logger.info("no HIGH-priority dives needing measurement")
        return None
