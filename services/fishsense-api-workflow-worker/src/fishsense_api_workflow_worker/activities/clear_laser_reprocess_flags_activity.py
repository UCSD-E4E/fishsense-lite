"""Lower a dive's stage-0.1 redraw flags once its JPEGs have been rewritten.

`needs_reprocess` is the one term in `select_next_for_laser_preprocessing`
that does not go false on its own. The rest of the cohort predicate drains
itself -- an image leaves the moment it has a label row -- but a flag stays
raised until something lowers it, so a parent firing that redrew the dive's
overlay JPEGs and left the flag set would re-select the same dive on the next
firing, re-stage its raw `.ORF`s from the NAS, and do it again every hour
while every higher-id dive waited behind it. Prod has been in that shape
before, without the flag: dive 60 held up dives 84/465/471 until 2026-08-04.

Called by `PreprocessLaserImagesParentWorkflow` after its data-worker child
completes, and only on that path -- a failed child leaves the flags up so the
dive is re-selected and genuinely retried.

Idempotent, which is what lets it run on every firing: a dive that was never
flagged clears zero rows and returns 0 rather than 404ing.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def clear_laser_reprocess_flags_activity(dive_id: int) -> int:
    """Clear `needs_reprocess` on every laser label of a dive's canonical
    images. Returns the number of rows cleared."""
    async with get_fs_client() as fs:
        cleared = await fs.labels.clear_laser_needs_reprocess(dive_id)

    activity.logger.info(
        "cleared laser reprocess flags dive_id=%d rows=%d", dive_id, cleared
    )
    return cleared
