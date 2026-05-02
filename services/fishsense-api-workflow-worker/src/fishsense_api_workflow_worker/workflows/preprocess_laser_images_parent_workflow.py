"""Stage 0.1 parent workflow (api-worker side).

Picks the next HIGH-priority dive needing laser preprocessing, resolves
its unlabeled-image-set + camera intrinsics via SDK, and dispatches
the resolved inputs to the data-worker's `PreprocessLaserImagesWorkflow`
on `fishsense_data_processing_queue`. After archive+cleanup, chains
into `PopulateLaserLabelStudioProjectWorkflow` on the api-worker so a
fresh dive lands in Label Studio in the same hourly run that produced
its JPEGs — no operator-triggered populate needed.

Cohort: HIGH-priority + at least one image with no completed
`LaserLabel` in any project. Mirrors the work-state shape of the
other three preprocess parents (dive-image / headtail / slate). The
earlier "no `LaserExtrinsics`" cohort tied stage 0.1 to a downstream
gate it doesn't actually advance, so dives whose laser side was done
but slate-side blocked stage-13 calibration kept getting re-selected
hourly with no work for the resolver to return — see
`select_next_for_laser_preprocessing` in the api's `dive_controller`.

Cluster-correctness invariants — relevant once the data-worker scales
beyond a single replica:

* Per-image activities (in the child workflow) PUT to nginx DAV which
  is idempotent under overwrite. Retried activities don't double-write.
* SDK upserts on the resolver side don't mutate state — read-only.
* The schedule that fires this workflow uses
  `overlap_policy=ScheduleOverlapPolicy.SKIP`, so a run still in
  flight when the next firing arrives is dropped at the schedule level.
* The data-worker child workflow is started with a deterministic id
  (`preprocess-laser-{dive_id}`) and
  `id_reuse_policy=ALLOW_DUPLICATE_FAILED_ONLY`; if a parent run
  somehow races past the schedule guard (manual + scheduled trigger
  overlap), or if a previous parent run failed *after* the child
  succeeded but *before* archive completed and the next firing
  re-targets the same dive, the second child dispatch hits
  `WorkflowAlreadyStarted` and the parent catches it. Archive +
  cleanup + populate then still run, so a child-then-parent split
  failure self-heals on the next firing rather than redoing
  per-image work.
* The populate child uses the same deterministic-id trick
  (`populate-laser-{dive_id}`). With the work-state cohort, dives
  drop out as labels complete, so re-firings on the same dive_id are
  the exception (resurrected re-incomplete labels, manual triggers)
  rather than the steady state — but the dedup still matters when
  they happen, since populate's task-import would otherwise create
  duplicate LS tasks for any image still flagged incomplete.
"""

from datetime import timedelta

from fishsense_shared import PreprocessLaserImagesInput
from temporalio import workflow
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.workflows._retry_policies import (
        SDK_FAIL_FAST_RETRY_POLICY,
    )

DATA_PROCESSING_TASK_QUEUE = "fishsense_data_processing_queue"
EXCHANGE_FOLDER = "preprocess_jpeg"
NAS_WORKFLOW = "laser"


@workflow.defn
class PreprocessLaserImagesParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive without laser extrinsics
    and dispatch its preprocessing to the data-worker.

    Returns the dive_id processed (or None when the backlog is empty).
    Each invocation drains exactly one dive — an N-dive backlog clears
    in N hourly schedule firings.
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await workflow.execute_activity(
            "select_next_high_priority_dive_for_laser_preprocessing_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
        )
        if dive_id is None:
            return None

        inputs = await workflow.execute_activity(
            "resolve_laser_preprocess_inputs_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
            result_type=PreprocessLaserImagesInput,
        )

        workflow.logger.info(
            "dispatching laser preprocess to data-worker dive_id=%d images=%d",
            inputs.dive_id,
            len(inputs.image_checksums),
        )

        if not inputs.image_checksums:
            return inputs.dive_id

        # Phase 3a: stage raw .ORF bytes from NAS to file-exchange
        # before the data-worker child runs. Failure here is fatal —
        # we don't want to dispatch a child that will 404 on every
        # download_raw. The next schedule firing retries; HEAD-check
        # in the staging activity makes the retry cheap for already-
        # staged checksums.
        await workflow.execute_activity(
            "stage_raw_bytes_for_dive_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(hours=1),
            heartbeat_timeout=timedelta(minutes=5),
        )

        try:
            await workflow.execute_child_workflow(
                "PreprocessLaserImagesWorkflow",
                inputs,
                id=f"preprocess-laser-{dive_id}",
                task_queue=DATA_PROCESSING_TASK_QUEUE,
                execution_timeout=timedelta(hours=1),
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
            )
        except WorkflowAlreadyStartedError:
            workflow.logger.info(
                "preprocess-laser-%d already ran successfully in a prior "
                "firing; skipping data-worker dispatch and continuing to "
                "archive + cleanup + populate",
                dive_id,
            )

        # Phase 3b: archive processed JPEGs to NAS, then drop the raw
        # `.ORF` bytes from the file-exchange. JPEGs intentionally stay
        # on the file-exchange — LS tasks reference them by URL and
        # their retention is a separate operational decision.
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
                "PopulateLaserLabelStudioProjectWorkflow",
                dive_id,
                id=f"populate-laser-{dive_id}",
                execution_timeout=timedelta(minutes=30),
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
            )
        except WorkflowAlreadyStartedError:
            workflow.logger.info(
                "populate-laser-%d already ran in a prior hourly firing; "
                "skipping LS task import",
                dive_id,
            )

        return inputs.dive_id
