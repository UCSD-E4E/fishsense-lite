"""Stage 0.1 parent workflow (api-worker side).

Picks the next HIGH-priority dive needing laser preprocessing, resolves
its unlabeled-image-set + camera intrinsics via SDK, and dispatches the
resolved inputs to the data-worker's `PreprocessLaserImagesWorkflow` on
`fishsense_data_processing_queue`.

Cohort: HIGH-priority + at least one canonical image with no non-sentinel
`LaserLabel` row (any real project), or one flagged `needs_reprocess`.
The earlier "no `LaserExtrinsics`"
cohort tied stage 0.1 to a downstream gate it doesn't advance, so dives
whose laser side was done but whose slate side blocked stage-13
calibration kept getting re-selected hourly with no work for the resolver
to return — see `select_next_for_laser_preprocessing` in the api's
`dive_cohort_controller`.

Laser populate is **decoupled** from this workflow (2026-07-28,
model-assisted labeling): it runs as its own scheduled parent,
`PopulateLaserLabelStudioProjectParentWorkflow` (+12 min), AFTER the
laser-detector predict stage (+10). Populate seeds non-sentinel
`LaserLabel` rows and the predict cohort excludes any image that already
has one, so populating here (at +0) would starve the predictor before it
ever ran.

`clear_laser_reprocess_flags_activity` runs last, and only on the path
where the child actually ran. It is what makes the `needs_reprocess`
cohort drainable: the flag is the one part of the stage-0.1 predicate that
does not go false on its own, so a firing that redrew the JPEGs and did
not lower it would re-select the same dive every hour and starve every
higher-id dive behind it. Appended after `cleanup_raw` rather than slotted
earlier so a child in flight at deploy still replays -- new commands after
the end of history are fine, a changed command in the middle is not.

The shared steps — and the cluster-correctness invariants behind each
(schedule SKIP overlap, deterministic child ids, ALLOW_DUPLICATE reuse,
idempotent per-image work) — live in `_dispatch`.
"""

from datetime import timedelta

from fishsense_shared import PreprocessLaserImagesInput
from temporalio import workflow

from fishsense_api_workflow_worker.workflows import _dispatch


@workflow.defn
class PreprocessLaserImagesParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive needing laser preprocessing
    and dispatch its work to the data-worker.

    Returns the dive_id processed (or None when the backlog is empty).
    Each invocation drains exactly one dive — an N-dive backlog clears
    in N hourly schedule firings.
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await _dispatch.select_dive(
            "select_next_high_priority_dive_for_laser_preprocessing_activity"
        )
        if dive_id is None:
            return None

        inputs = await _dispatch.resolve_inputs(
            "resolve_laser_preprocess_inputs_activity",
            dive_id,
            PreprocessLaserImagesInput,
        )

        workflow.logger.info(
            "dispatching laser preprocess to data-worker dive_id=%d images=%d",
            inputs.dive_id,
            len(inputs.image_checksums),
        )

        if not inputs.image_checksums:
            return inputs.dive_id

        await _dispatch.wake_data_worker()
        await _dispatch.stage_raw(dive_id)
        await _dispatch.dispatch_child(
            "PreprocessLaserImagesWorkflow",
            inputs,
            child_id=f"preprocess-laser-{dive_id}",
            execution_timeout=timedelta(hours=1),
        )
        await _dispatch.cleanup_raw(dive_id)
        await _dispatch.run_sdk_activity(
            "clear_laser_reprocess_flags_activity", dive_id
        )

        return inputs.dive_id
