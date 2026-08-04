"""Stage 13 parent workflow (api-worker side).

Picks the next HIGH-priority dive needing laser calibration and
dispatches it to the data-worker's `PerformLaserCalibrationWorkflow`
on `fishsense_data_processing_queue`.

Lighter than the four preprocess parents (0.1 / 2 / 5.1 / 9): no NAS
staging, no file-exchange JPEGs, no per-image fan-out. Calibration is
pure SDK math against already-stored slate + laser labels + camera
intrinsics. The activity itself does its own SDK fetches inline (per
CLAUDE.md, stages 13 and 14 deliberately keep SDK fetches in the
data-worker because the math kernels need fishsense-core anyway).

Cluster-correctness invariants — relevant once the data-worker scales
beyond a single replica:

* The schedule that fires this workflow uses
  `overlap_policy=ScheduleOverlapPolicy.SKIP`, so a run still in
  flight when the next firing arrives is dropped at the schedule level.
* The child workflow is started with a deterministic id
  (`perform-laser-calibration-{dive_id}`) and
  `id_reuse_policy=ALLOW_DUPLICATE` (changed from FAILED_ONLY on
  2026-08-04, mirroring the preprocess children's 2026-07-23 flip:
  FAILED_ONLY means a *completed* child id can never re-dispatch, so
  a dive whose bad calibration was remediated — labels cleaned, stale
  extrinsics row deleted — could never be refit through the parent;
  the re-dispatch was swallowed silently. Observed live on dive 77,
  2026-08-04.) ALLOW_DUPLICATE is safe: the cohort only offers dives
  with no extrinsics row, `put_laser_extrinsics` upserts on dive_id,
  and the activity's self-consistency gate refuses to persist a fit
  that disagrees with the dive's own laser dots.
  `WorkflowAlreadyStartedError` is now only raised while a prior
  child with the same id is still *running*; the parent catches it.
* `put_laser_extrinsics` on the activity side is an upsert — even
  outright re-runs of a successful call land on the same row.
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.workflows._retry_policies import (
        SCALING_RETRY_POLICY,
        SDK_FAIL_FAST_RETRY_POLICY,
    )

DATA_PROCESSING_TASK_QUEUE = "fishsense_data_processing_queue"


@workflow.defn
class PerformLaserCalibrationParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive lacking laser extrinsics
    and dispatch its calibration to the data-worker.

    Returns the dive_id processed (or None when the backlog is empty).
    Each invocation drains exactly one dive — an N-dive backlog clears
    in N hourly schedule firings.
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await workflow.execute_activity(
            "select_next_high_priority_dive_for_laser_calibration_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
        )
        if dive_id is None:
            return None

        workflow.logger.info(
            "dispatching laser calibration to data-worker dive_id=%d", dive_id
        )

        # Wake the NRP data-worker before its child workflow lands on
        # the queue (it scales to zero when idle). Idempotent — converges
        # on the configured replica count, never accumulates; a no-op
        # when k8s scaling isn't configured.
        await workflow.execute_activity(
            "ensure_data_worker_running_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SCALING_RETRY_POLICY,
        )

        try:
            await workflow.execute_child_workflow(
                "PerformLaserCalibrationWorkflow",
                dive_id,
                id=f"perform-laser-calibration-{dive_id}",
                task_queue=DATA_PROCESSING_TASK_QUEUE,
                execution_timeout=timedelta(minutes=15),
                # ALLOW_DUPLICATE (not FAILED_ONLY): a completed child id
                # must not block a refit after remediation (bad extrinsics
                # deleted, labels cleaned). Safe: cohort = no-extrinsics
                # dives only, the PUT upserts on dive_id, and the activity's
                # self-consistency gate rejects fits that disagree with the
                # dive's own dots. See module docstring.
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
            )
        except WorkflowAlreadyStartedError:
            # Only reachable while a prior child with this id is still
            # RUNNING — a completed child no longer blocks.
            workflow.logger.info(
                "perform-laser-calibration-%d is still running; "
                "skipping duplicate dispatch",
                dive_id,
            )

        return dive_id
