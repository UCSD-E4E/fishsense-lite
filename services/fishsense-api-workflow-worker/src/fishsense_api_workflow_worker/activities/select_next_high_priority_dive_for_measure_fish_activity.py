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

The selector is a single SDK call; the SQL predicate lives in the
api's `select-next/measure-fish` endpoint.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def select_next_high_priority_dive_for_measure_fish_activity() -> int | None:
    async with get_fs_client() as fs:
        dive_id = await fs.dives.select_next_for_measure_fish()

    if dive_id is None:
        activity.logger.info("no HIGH-priority dives needing measurement")
    else:
        activity.logger.info(
            "next HIGH-priority dive needing measurement: dive_id=%d",
            dive_id,
        )
    return dive_id
