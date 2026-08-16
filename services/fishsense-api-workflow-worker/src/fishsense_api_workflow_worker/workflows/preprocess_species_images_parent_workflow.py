"""Stage 2 parent workflow (api-worker side).

Picks the next HIGH-priority dive needing species preprocessing, resolves
its PREDICTION clusters + camera intrinsics + laser/species labels via
SDK, and dispatches `PreprocessSpeciesImagesWorkflow` to the data-worker.
The child writes the group-preprocessed JPEGs to Garage; this parent then
evicts the staged raw scratch and stops.

Species LS-task population is **decoupled** from this workflow — it does
not chain into `PopulateSpeciesLabelStudioProjectWorkflow`. The hourly
`PopulateSpeciesLabelStudioProjectParentWorkflow` (+20 min) selects the
superseded-aware "needs species population" cohort and fans out the
idempotent, JPEG-gated populate per dive. Decoupling lets dives whose
old-project species rows were superseded (post hosted-LS migration) get
(re)populated without re-preprocessing.

Shared steps live in `_dispatch`; see `PreprocessLaserImagesParentWorkflow`
and CLAUDE.md's "Cross-worker orchestration pattern" for the invariants.
"""

from datetime import timedelta

from fishsense_shared import PreprocessSpeciesImagesInput
from temporalio import workflow

from fishsense_api_workflow_worker.workflows import _dispatch


@workflow.defn
class PreprocessSpeciesImagesParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive needing species
    preprocessing and dispatch its work to the data-worker.

    Returns the dive_id processed (or None when the backlog is empty).
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await _dispatch.select_dive(
            "select_next_high_priority_dive_for_species_preprocessing_activity"
        )
        if dive_id is None:
            return None

        inputs = await _dispatch.resolve_inputs(
            "resolve_species_preprocess_inputs_activity",
            dive_id,
            PreprocessSpeciesImagesInput,
        )

        total_images = sum(len(cluster) for cluster in inputs.clusters)
        workflow.logger.info(
            "dispatching species preprocess to data-worker dive_id=%d "
            "clusters=%d images=%d",
            inputs.dive_id,
            len(inputs.clusters),
            total_images,
        )

        if not inputs.clusters or total_images == 0:
            return inputs.dive_id

        await _dispatch.wake_data_worker()
        await _dispatch.stage_raw(dive_id)
        await _dispatch.dispatch_child(
            "PreprocessSpeciesImagesWorkflow",
            inputs,
            child_id=f"preprocess-species-{dive_id}",
            execution_timeout=timedelta(hours=2),
        )
        await _dispatch.cleanup_raw(dive_id)

        return inputs.dive_id
