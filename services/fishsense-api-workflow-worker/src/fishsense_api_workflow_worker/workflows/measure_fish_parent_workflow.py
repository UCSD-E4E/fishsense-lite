"""Stage 14 parent workflow (api-worker side).

Picks the next HIGH-priority dive ready for fish measurement and
dispatches it to the data-worker's `MeasureFishWorkflow` on
`fishsense_data_processing_queue`.

Lighter than the four preprocess parents (0.1 / 2 / 5.1 / 9): no NAS
staging, no file-exchange JPEGs, no per-image fan-out. Measurement is
pure SDK math against already-stored laser/headtail/cluster data. The
data-worker activity does its own SDK fetches inline (per CLAUDE.md,
stages 13 and 14 keep SDK fetches in the data-worker because the math
kernels need fishsense-core anyway).

**Scheduled hourly at +40 min** (2026-07-17). It was operator-triggered
for a long time because `measure_fish_activity` was non-idempotent
(`post_measurement` was a plain POST with no per-image filter), so a
re-run on a partially-measured dive duplicated measurements — and
because the cohort predicate never went false, a schedule would have
re-measured the same dives every hour forever. Both are fixed:
measurement upserts on `(image_id, fish_id)`, the activity skips
already-measured images, and the cohort is scoped to images that can
actually be measured, so it drains to empty.

Still runnable on demand for backfill; use a non-colliding workflow id
so the schedule's own id stays free:

```
temporal workflow start \
    --task-queue fishsense_api_queue \
    --type MeasureFishParentWorkflow \
    --workflow-id measure-fish-parent-<run-tag>
```

The selector activity returns the next eligible dive_id (HIGH priority
+ has laser_extrinsics + at least one *measurable* image with no
`Measurement` — "measurable" being a top-three species label whose image
has a valid laser label, a valid head/tail label and a LABEL_STUDIO
cluster). The parent then dispatches the child and returns the dive_id
processed. Each run drains exactly one dive, so a backlog clears one
dive per hour.

Cluster-correctness invariants — relevant once the data-worker scales
beyond a single replica:

* The child workflow is started with a deterministic id
  (`measure-fish-{dive_id}`) and
  `id_reuse_policy=ALLOW_DUPLICATE_FAILED_ONLY`; a duplicate parent
  run targeting the same dive hits `WorkflowAlreadyStarted` (whether
  the prior child is still running or completed successfully) and
  the parent catches it. This is now a don't-redo-work guard rather
  than a correctness one — a duplicate dispatch would be harmless
  since measurement is idempotent at both the write layer and in the
  activity's skip.
* Per-cluster `fish_id` rebinds via `put_cluster` on the activity side
  are idempotent under retry within a single child run.
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
class MeasureFishParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive ready for measurement and
    dispatch its child workflow to the data-worker.

    Returns the dive_id processed (or None when the cohort is empty).
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await workflow.execute_activity(
            "select_next_high_priority_dive_for_measure_fish_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
        )
        if dive_id is None:
            return None

        workflow.logger.info(
            "dispatching fish measurement to data-worker dive_id=%d", dive_id
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
                "MeasureFishWorkflow",
                dive_id,
                id=f"measure-fish-{dive_id}",
                task_queue=DATA_PROCESSING_TASK_QUEUE,
                execution_timeout=timedelta(hours=1),
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
            )
        except WorkflowAlreadyStartedError:
            # Not a safety gate any more — measurement is idempotent as of
            # 2026-07-17 (`post_measurement` upserts on (image_id, fish_id)
            # and the activity skips already-measured images), so a
            # duplicate dispatch would be harmless. This just avoids
            # re-doing work already done for this dive.
            workflow.logger.info(
                "measure-fish-%d already ran successfully; skipping "
                "duplicate dispatch (no re-work needed)",
                dive_id,
            )

        return dive_id
