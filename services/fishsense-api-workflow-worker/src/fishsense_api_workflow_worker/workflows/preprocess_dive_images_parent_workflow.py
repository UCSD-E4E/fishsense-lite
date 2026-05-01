"""Stage 2 parent workflow (api-worker side).

Picks the next HIGH-priority dive needing dive-image preprocessing,
resolves its PREDICTION clusters + camera intrinsics via SDK, and
dispatches `PreprocessDiveImagesWorkflow` as a child on the
data-worker (`fishsense_data_processing_queue`).

Same cluster-correctness invariants as
`PreprocessLaserImagesParentWorkflow` — see CLAUDE.md's "Cross-worker
orchestration pattern" section.
"""

from datetime import timedelta

from fishsense_shared import PreprocessDiveImagesInput
from temporalio import workflow

DATA_PROCESSING_TASK_QUEUE = "fishsense_data_processing_queue"


@workflow.defn
class PreprocessDiveImagesParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive needing dive-image
    preprocessing and dispatch its work to the data-worker.

    Returns the dive_id processed (or None when the backlog is empty).
    Each invocation drains exactly one dive — an N-dive backlog clears
    in N hourly schedule firings.
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await workflow.execute_activity(
            "select_next_high_priority_dive_for_dive_image_preprocessing_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
        )
        if dive_id is None:
            return None

        inputs = await workflow.execute_activity(
            "resolve_dive_image_preprocess_inputs_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
            result_type=PreprocessDiveImagesInput,
        )

        total_images = sum(len(cluster) for cluster in inputs.clusters)
        workflow.logger.info(
            "dispatching dive-image preprocess to data-worker dive_id=%d "
            "clusters=%d images=%d",
            inputs.dive_id,
            len(inputs.clusters),
            total_images,
        )

        if not inputs.clusters or total_images == 0:
            return inputs.dive_id

        await workflow.execute_child_workflow(
            "PreprocessDiveImagesWorkflow",
            inputs,
            id=f"preprocess-dive-images-{dive_id}",
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            execution_timeout=timedelta(hours=2),
        )

        return inputs.dive_id
