"""Lower a dive's stage 9 redraw flags once its JPEGs have been rewritten.

`needs_reprocess` is the one term in the stage 9 cohort predicate that does not
go false on its own. The rest drains itself -- an image leaves the moment it has
a label row -- but a flag stays raised until something lowers it, so a parent
firing that redrew the dive's JPEGs and left the flag set would re-select the
same dive on the next firing, re-stage its raw `.ORF`s from the NAS, and do it
again every hour while every higher-id dive waited behind it. Prod has been in
that shape before, without the flag: dive 60 held up dives 84/465/471 until
2026-08-04.

Called by the stage 9 parent after its data-worker child completes, and only on
that path -- a failed child leaves the flags up so the dive is re-selected and
genuinely retried.

One activity per stage rather than one generic activity taking a stage name.
Temporal registers and records activities *by name*, so per-stage names are
what makes a workflow history readable ("which stage cleared what"), and
`_dispatch.run_sdk_activity` passes exactly one argument. The four bodies are
four lines each and differ only in the SDK call, which `duplicate-code` cannot
see (it is textual, and `clear_species_needs_reprocess` and
`clear_headtail_needs_reprocess` are different strings) -- so this is recorded
here deliberately rather than left for a reviewer to wonder about.

Idempotent, which is what lets it run on every firing: a dive that was never
flagged clears zero rows and returns 0 rather than 404ing.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def clear_slate_reprocess_flags_activity(dive_id: int) -> int:
    """Clear `needs_reprocess` on every dive-slate label of a dive's canonical
    images. Returns the number of rows cleared."""
    async with get_fs_client() as fs:
        cleared = await fs.labels.clear_dive_slate_needs_reprocess(dive_id)

    activity.logger.info(
        "cleared slate reprocess flags dive_id=%d rows=%d", dive_id, cleared
    )
    return cleared
