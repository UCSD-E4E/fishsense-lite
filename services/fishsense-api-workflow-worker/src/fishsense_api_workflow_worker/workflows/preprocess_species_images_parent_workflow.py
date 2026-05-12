"""Stage 2 parent workflow (api-worker side).

Picks the next HIGH-priority dive needing species preprocessing,
resolves its PREDICTION clusters + camera intrinsics + laser/species
labels via SDK, and dispatches `PreprocessSpeciesImagesWorkflow` as a
child on the data-worker (`fishsense_data_processing_queue`). After
archive+cleanup, chains into `PopulateSpeciesLabelStudioProjectWorkflow`
so the group-preprocessed JPEGs land in the per-dive species LS
project in the same hourly run that produced them.

Same cluster-correctness invariants as
`PreprocessLaserImagesParentWorkflow` — see CLAUDE.md's "Cross-worker
orchestration pattern" section. The populate child uses a
deterministic id (`populate-species-{dive_id}`) so subsequent
re-firings on the same cohort dive no-op via WorkflowAlreadyStarted
rather than re-importing duplicate LS tasks.
"""

from datetime import timedelta

from fishsense_shared import PreprocessSpeciesImagesInput
from temporalio import workflow
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.workflows._retry_policies import (
        SCALING_RETRY_POLICY,
        SDK_FAIL_FAST_RETRY_POLICY,
    )

DATA_PROCESSING_TASK_QUEUE = "fishsense_data_processing_queue"
EXCHANGE_FOLDER = "preprocess_groups_jpeg"
NAS_WORKFLOW = "species_images"


@workflow.defn
class PreprocessSpeciesImagesParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive needing species
    preprocessing and dispatch its work to the data-worker.

    Returns the dive_id processed (or None when the backlog is empty).
    Each invocation drains exactly one dive — an N-dive backlog clears
    in N hourly schedule firings.
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await workflow.execute_activity(
            "select_next_high_priority_dive_for_species_preprocessing_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
        )
        if dive_id is None:
            return None

        inputs = await workflow.execute_activity(
            "resolve_species_preprocess_inputs_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
            result_type=PreprocessSpeciesImagesInput,
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

        # Wake the NRP data-worker before its child workflow lands on
        # the queue (it scales to zero when idle). Idempotent — converges
        # on the configured replica count, never accumulates; a no-op
        # when k8s scaling isn't configured. Returns immediately, so the
        # pod's cold start overlaps the NAS-staging step below.
        await workflow.execute_activity(
            "ensure_data_worker_running_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SCALING_RETRY_POLICY,
        )

        await workflow.execute_activity(
            "stage_raw_bytes_for_dive_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(hours=1),
            heartbeat_timeout=timedelta(minutes=5),
        )

        try:
            await workflow.execute_child_workflow(
                "PreprocessSpeciesImagesWorkflow",
                inputs,
                id=f"preprocess-species-{dive_id}",
                task_queue=DATA_PROCESSING_TASK_QUEUE,
                execution_timeout=timedelta(hours=2),
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
            )
        except WorkflowAlreadyStartedError:
            workflow.logger.info(
                "preprocess-species-%d already ran successfully in a "
                "prior firing; skipping data-worker dispatch and continuing "
                "to archive + cleanup + populate",
                dive_id,
            )

        await workflow.execute_activity(
            "archive_processed_jpegs_to_nas_activity",
            args=(dive_id, EXCHANGE_FOLDER, NAS_WORKFLOW),
            schedule_to_close_timeout=timedelta(hours=1),
            heartbeat_timeout=timedelta(minutes=5),
        )
        await workflow.execute_activity(
            "cleanup_raw_bytes_for_dive_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=15),
            heartbeat_timeout=timedelta(minutes=5),
        )

        try:
            await workflow.execute_child_workflow(
                "PopulateSpeciesLabelStudioProjectWorkflow",
                dive_id,
                id=f"populate-species-{dive_id}",
                execution_timeout=timedelta(minutes=30),
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
            )
        except WorkflowAlreadyStartedError:
            workflow.logger.info(
                "populate-species-%d already ran in a prior hourly firing; "
                "skipping LS task import",
                dive_id,
            )

        return inputs.dive_id
