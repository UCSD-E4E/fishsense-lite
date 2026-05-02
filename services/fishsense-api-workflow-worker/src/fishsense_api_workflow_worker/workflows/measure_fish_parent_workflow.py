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

**Deliberately not scheduled.** `measure_fish_activity` is non-idempotent
(`post_measurement` is a POST and the SDK has no per-image measurement
query), so a re-run on a partially-failed dive will duplicate
measurements on already-bound clusters. Until that's resolved
(probably by adding an SDK get-measurements query and per-image
filtering in the activity), this parent is invoked on-demand:

```
temporal workflow start \
    --task-queue fishsense_api_queue \
    --type MeasureFishParentWorkflow \
    --workflow-id measure-fish-parent-<run-tag>
```

The selector activity returns the next eligible dive_id (HIGH priority
+ has laser_extrinsics + has LABEL_STUDIO clusters with at least one
unbound `fish_id`); the parent then dispatches the child and returns
the dive_id processed.

Cluster-correctness invariants — relevant once the data-worker scales
beyond a single replica:

* The child workflow is started with a deterministic id
  (`measure-fish-{dive_id}`); a duplicate parent run targeting the
  same dive hits `WorkflowAlreadyStarted` and is a no-op while the
  first child is still running.
* Per-cluster `fish_id` rebinds via `put_cluster` on the activity side
  are idempotent under retry within a single child run.
"""

from datetime import timedelta

from temporalio import workflow

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
        )
        if dive_id is None:
            return None

        workflow.logger.info(
            "dispatching fish measurement to data-worker dive_id=%d", dive_id
        )

        await workflow.execute_child_workflow(
            "MeasureFishWorkflow",
            dive_id,
            id=f"measure-fish-{dive_id}",
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            execution_timeout=timedelta(hours=1),
        )

        return dive_id
