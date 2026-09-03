"""Stage 5.1 parent workflow (api-worker side).

Picks the next HIGH-priority dive needing head/tail preprocessing and
dispatches `PreprocessHeadtailImagesWorkflow` to the data-worker. After
cleanup it chains into `PopulateHeadTailLabelStudioProjectWorkflow`, so
head/tail JPEGs land in their LS project in the same hourly firing that
produced them.

Cohort cascades from *valid laser labels* (flipped 2026-05-04), not from
`SpeciesLabel.top_three_photos_of_group` — head/tail work starts as soon
as the laser labelers and the validator have signed off on an image,
in parallel with stages 1/2.

Shared steps live in `_dispatch`; see `PreprocessLaserImagesParentWorkflow`
and CLAUDE.md for the cluster-correctness invariants.
"""

from datetime import timedelta

from fishsense_shared import PreprocessHeadtailImagesInput
from temporalio import workflow

from fishsense_api_workflow_worker.workflows import _dispatch


@workflow.defn
class PreprocessHeadtailImagesParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive needing head/tail
    preprocessing and dispatch its work to the data-worker.

    Returns the dive_id processed (or None when the backlog is empty).
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await _dispatch.select_dive(
            "select_next_high_priority_dive_for_headtail_preprocessing_activity"
        )
        if dive_id is None:
            return None

        inputs = await _dispatch.resolve_inputs(
            "resolve_headtail_preprocess_inputs_activity",
            dive_id,
            PreprocessHeadtailImagesInput,
        )

        workflow.logger.info(
            "dispatching headtail preprocess to data-worker dive_id=%d images=%d",
            inputs.dive_id,
            len(inputs.image_checksums),
        )

        if not inputs.image_checksums:
            return inputs.dive_id

        await _dispatch.wake_data_worker()
        await _dispatch.stage_raw(dive_id)
        await _dispatch.dispatch_child(
            "PreprocessHeadtailImagesWorkflow",
            inputs,
            child_id=f"preprocess-headtail-{dive_id}",
            execution_timeout=timedelta(hours=1),
        )
        await _dispatch.cleanup_raw(dive_id)
        # Populate is NOT dispatched here any more. It moved to its own hourly
        # parent at +34, after the +32 head/tail predict parent, because
        # populate seeds sentinel `HeadTailLabel` rows and the predict cohort
        # excludes any image carrying a live label — chaining populate straight
        # off preprocess would starve every image of a prediction permanently.
        # Same decoupling, for the same reason, that the laser stage did at
        # +10/+12. See `PopulateHeadTailLabelStudioProjectParentWorkflow`.

        return inputs.dive_id
