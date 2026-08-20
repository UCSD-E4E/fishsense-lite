"""Laser-depth parent workflow (api-worker side).

Picks the next HIGH-priority dive whose laser depths are missing or stale and
dispatches it to the data-worker's `ComputeLaserDepthsWorkflow` on
`fishsense_data_processing_queue`.

Structurally the lightest parent in the tree, alongside stages 13 and 14: no
NAS staging, no object-store traffic, no per-image fan-out. The work is pure
SDK math over labels the pipeline already holds — it lives on the data-worker
only because the projection kernel is in fishsense-core.

**Scheduled hourly at +25 min**, between the species-populate parent (+20) and
the headtail preprocess parent (+30). Ordering against stage 13 (+50) does not
matter: a dive calibrated at :50 gets its depths at :25 the next hour, which
is irrelevant at this cadence, and slotting after calibration would crowd the
+55 scale-down sweeper.

The cohort is keyed on the inputs a depth is derived from — the laser label
and the resolved calibration — rather than on whether a row exists. So it
drains like any other stage, and it re-enters a dive by itself when a
recalibration or a relabel invalidates what was stored, which is what makes
the initial backfill just "the first few firings" rather than a script someone
has to run and then remember.

Still runnable on demand; use a non-colliding workflow id so the schedule's
own id stays free:

```
temporal workflow start \
    --task-queue fishsense_api_queue \
    --type ComputeLaserDepthsParentWorkflow \
    --workflow-id compute-laser-depths-parent-<run-tag>
```

Each invocation drains exactly one dive — call it repeatedly to clear a
backlog.

Cluster-correctness invariants, mirroring the other dispatch parents:

* Deterministic child id (`compute-laser-depths-{dive_id}`) with
  `id_reuse_policy=ALLOW_DUPLICATE`. FAILED_ONLY would burn the id on the
  first successful run, so a dive could never have its depths recomputed
  after a recalibration — the same trap that stranded prod dives 59/439 in
  preprocess and dive 60 in populate. ALLOW_DUPLICATE is safe here because
  the activity is idempotent: it skips images already current and
  `put_laser_depth` upserts on `image_id`.
* `WorkflowAlreadyStartedError` is therefore only raised while a prior child
  for that dive is still *running* (a manual run overlapping the schedule),
  and is caught.
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
class ComputeLaserDepthsParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive needing laser depths and dispatch
    its child workflow to the data-worker.

    Returns the dive_id processed (or None when the cohort is empty).
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await workflow.execute_activity(
            "select_next_high_priority_dive_for_laser_depth_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
        )
        if dive_id is None:
            return None

        workflow.logger.info(
            "dispatching laser depth computation to data-worker dive_id=%d", dive_id
        )

        # Wake the NRP data-worker before its child lands on the queue (it
        # scales to zero when idle). Idempotent — converges on the configured
        # replica count; a no-op when k8s scaling isn't configured.
        await workflow.execute_activity(
            "ensure_data_worker_running_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SCALING_RETRY_POLICY,
        )

        try:
            await workflow.execute_child_workflow(
                "ComputeLaserDepthsWorkflow",
                dive_id,
                id=f"compute-laser-depths-{dive_id}",
                task_queue=DATA_PROCESSING_TASK_QUEUE,
                execution_timeout=timedelta(hours=1),
                # ALLOW_DUPLICATE, not FAILED_ONLY: a completed child id must
                # not block recomputation after a recalibration. See module
                # docstring.
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
            )
        except WorkflowAlreadyStartedError:
            # Only reachable while a prior child with this id is still
            # RUNNING — a completed one no longer blocks under
            # ALLOW_DUPLICATE.
            workflow.logger.info(
                "compute-laser-depths-%d is still running; skipping duplicate "
                "dispatch",
                dive_id,
            )

        return dive_id
