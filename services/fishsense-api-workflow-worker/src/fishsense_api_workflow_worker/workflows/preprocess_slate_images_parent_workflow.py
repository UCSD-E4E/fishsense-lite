"""Stage 9 parent workflow (api-worker side).

Picks the next HIGH-priority dive needing slate preprocessing and
dispatches `PreprocessSlateImagesWorkflow` to the data-worker. This stage
stages **two** inputs — the raw `.ORF` frames and the dive's slate
template PDF — because the per-image activity composites the rendered
template alongside the rectified frame. After cleanup it chains into
`PopulateDiveSlateLabelStudioProjectWorkflow`.

Cohort: HIGH-priority + `dive_slate_id` set + at least one
`SpeciesLabel.content_of_image = 'Slate, Laser on slate'` whose image
carries no `DiveSlateLabel` row at all.

Shared steps live in `_dispatch`; see `PreprocessLaserImagesParentWorkflow`
and CLAUDE.md for the cluster-correctness invariants.
"""

from datetime import timedelta

from fishsense_shared import PreprocessSlateImagesInput
from temporalio import workflow

from fishsense_api_workflow_worker.workflows import _dispatch


@workflow.defn
class PreprocessSlateImagesParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive needing slate preprocessing
    and dispatch its work to the data-worker.

    Returns the dive_id processed (or None when the backlog is empty).
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await _dispatch.select_dive(
            "select_next_high_priority_dive_for_slate_preprocessing_activity"
        )
        if dive_id is None:
            return None

        inputs = await _dispatch.resolve_inputs(
            "resolve_slate_preprocess_inputs_activity",
            dive_id,
            PreprocessSlateImagesInput,
        )

        workflow.logger.info(
            "dispatching slate preprocess to data-worker dive_id=%d images=%d slate_id=%d",
            inputs.dive_id,
            len(inputs.image_checksums),
            inputs.slate_id,
        )

        if not inputs.image_checksums:
            return inputs.dive_id

        await _dispatch.wake_data_worker()
        await _dispatch.stage_raw(dive_id)
        await _dispatch.stage_slate_pdf(
            inputs.slate_id,
            schedule_to_close_timeout=timedelta(minutes=5),
        )
        await _dispatch.dispatch_child(
            "PreprocessSlateImagesWorkflow",
            inputs,
            child_id=f"preprocess-slate-{dive_id}",
            execution_timeout=timedelta(hours=1),
        )
        await _dispatch.cleanup_raw(dive_id)
        await _dispatch.dispatch_populate(
            "PopulateDiveSlateLabelStudioProjectWorkflow",
            dive_id,
            f"populate-dive-slate-{dive_id}",
        )

        return inputs.dive_id
