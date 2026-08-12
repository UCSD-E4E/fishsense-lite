"""Shared steps for the cross-worker dispatch parents.

Six parent workflows (preprocess laser / species / headtail / slate,
predict laser / slate) run the same play: pick a dive, resolve its
inputs, wake the data-worker, stage its bytes, hand a child workflow to
`fishsense_data_processing_queue`, then clean up. They were copy-pasted
from each other, and the copies drifted — comments in the laser parent
described an `ALLOW_DUPLICATE_FAILED_ONLY` policy its own code hadn't
used for months, and referred to a populate child that had been
decoupled. Each step lives here once so there is one place to read, and
one place to change.

**These helpers emit the same Temporal commands, in the same order, that
the inlined code did.** That is deliberate and load-bearing: a workflow's
command sequence is its replay contract, so a run in flight when this
ships still replays cleanly. Timeouts and retry policies are therefore
parameters rather than constants — each stage keeps the exact values it
already had, even where they look arbitrary (slate-PDF staging uses
5 min in the preprocess parent and 15 min in the predict parent).

Keep these as thin, individually-callable steps rather than one
`run_parent()` that takes a dozen flags. Each parent stays a readable
top-to-bottom narrative of what it does — which is how workflow code
wants to read — while the boilerplate lives in one place.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy, WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.workflows._retry_policies import (
        SCALING_RETRY_POLICY,
        SDK_FAIL_FAST_RETRY_POLICY,
        STAGE_RAW_RETRY_POLICY,
    )

DATA_PROCESSING_TASK_QUEUE = "fishsense_data_processing_queue"

__all__ = [
    "DATA_PROCESSING_TASK_QUEUE",
    "STAGE_RAW_RETRY_POLICY",
    "cleanup_raw",
    "dispatch_child",
    "dispatch_populate",
    "run_sdk_activity",
    "resolve_inputs",
    "select_dive",
    "stage_raw",
    "stage_slate_pdf",
    "wake_data_worker",
]


async def select_dive(activity_name: str) -> int | None:
    """Run a cohort selector. Returns the next dive_id, or None when the
    cohort is empty and this firing has nothing to do."""
    return await workflow.execute_activity(
        activity_name,
        args=(),
        schedule_to_close_timeout=timedelta(minutes=5),
        retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
    )


async def resolve_inputs(activity_name: str, dive_id: int, result_type: type) -> Any:
    """Run a resolver, returning the fully-populated workflow-input DTO
    the data-worker child consumes."""
    return await workflow.execute_activity(
        activity_name,
        args=(dive_id,),
        schedule_to_close_timeout=timedelta(minutes=5),
        retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
        result_type=result_type,
    )


async def wake_data_worker() -> None:
    """Scale the NRP data-worker up before its child lands on the queue
    (it scales to zero when idle).

    Idempotent — converges on the configured replica count, never
    accumulates; a no-op when k8s scaling isn't configured. Returns
    immediately, so the pod's cold start overlaps the staging steps.
    """
    await workflow.execute_activity(
        "ensure_data_worker_running_activity",
        args=(),
        schedule_to_close_timeout=timedelta(minutes=5),
        retry_policy=SCALING_RETRY_POLICY,
    )


async def stage_raw(dive_id: int) -> None:
    """Copy the dive's raw `.ORF` bytes from NAS into Garage scratch.

    Failure here is fatal on purpose — dispatching a child that would 404
    on every `download_raw` wastes a full fan-out. The next schedule
    firing retries; the staging activity's HEAD-check makes that cheap for
    checksums already staged.
    """
    await workflow.execute_activity(
        "stage_raw_bytes_for_dive_activity",
        args=(dive_id,),
        schedule_to_close_timeout=timedelta(hours=1),
        heartbeat_timeout=timedelta(minutes=5),
        retry_policy=STAGE_RAW_RETRY_POLICY,
    )


async def stage_slate_pdf(
    slate_id: int,
    *,
    schedule_to_close_timeout: timedelta,
    heartbeat_timeout: timedelta | None = None,
    retry_policy: RetryPolicy | None = None,
) -> None:
    """Stage the slate template PDF (stages 9 and slate-predict).

    The two callers pass different timeouts — preserved as parameters
    rather than unified, so this refactor changes no in-flight run's
    replay contract. Unifying them is a separate, deliberate change.
    """
    await workflow.execute_activity(
        "stage_slate_pdf_activity",
        args=(slate_id,),
        schedule_to_close_timeout=schedule_to_close_timeout,
        heartbeat_timeout=heartbeat_timeout,
        retry_policy=retry_policy,
    )


async def dispatch_child(
    workflow_name: str,
    inputs: Any,
    *,
    child_id: str,
    execution_timeout: timedelta,
    result_type: type | None = None,
) -> Any:
    """Start the data-worker child and wait for it.

    The child id is deterministic (`preprocess-laser-{dive_id}`) and the
    reuse policy is **ALLOW_DUPLICATE**, not `ALLOW_DUPLICATE_FAILED_ONLY`.
    FAILED_ONLY meant a *completed* id could never re-dispatch, so a dive
    could never be reprocessed to pick up images that became processable
    after its first successful run — a laser validated after one-shot
    stage-1 clustering, or an orphan later assigned a cluster. Those
    images' JPEGs were never produced and populate deferred them forever
    (prod dives 59/439). ALLOW_DUPLICATE is safe here: resolvers return
    only images that still need work, and the per-image activities are
    idempotent (S3 overwrite, SDK upsert).

    With ALLOW_DUPLICATE, `WorkflowAlreadyStartedError` is reachable only
    while a prior child with this id is still *running* (a manual run
    overlapping the schedule). That run is doing the work, so the caller
    continues to its cleanup rather than failing the firing. Returns None
    in that case.
    """
    try:
        return await workflow.execute_child_workflow(
            workflow_name,
            inputs,
            id=child_id,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            execution_timeout=execution_timeout,
            id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
            result_type=result_type,
        )
    except WorkflowAlreadyStartedError:
        workflow.logger.info(
            "%s is still running; skipping duplicate dispatch and continuing",
            child_id,
        )
        return None


async def run_sdk_activity(activity_name: str, arg: Any) -> None:
    """Run a single-argument SDK write-back step (15 min, fail-fast).

    Covers the predict parents' post-child work: persisting the child's
    returned rows (`persist_*_predictions_activity`, arg = results) and
    attaching them to existing LS tasks
    (`backfill_slate_predictions_for_dive_activity`, arg = dive_id).
    Both are idempotent, so the fail-fast policy is safe — a failure just
    means the next schedule firing redoes it.
    """
    await workflow.execute_activity(
        activity_name,
        args=(arg,),
        schedule_to_close_timeout=timedelta(minutes=15),
        retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
    )


async def cleanup_raw(dive_id: int) -> None:
    """Evict the dive's staged raw `.ORF` scratch from Garage.

    The JPEGs are the durable artifact and stay (Label Studio reads them
    via presign); only the reproducible-from-NAS scratch is dropped. The
    NAS source is never touched — see the tripwire test asserting the
    cleanup module imports no NAS client.
    """
    await workflow.execute_activity(
        "cleanup_raw_bytes_for_dive_activity",
        args=(dive_id,),
        schedule_to_close_timeout=timedelta(minutes=15),
        heartbeat_timeout=timedelta(minutes=5),
    )


async def dispatch_populate(workflow_name: str, dive_id: int, child_id: str) -> None:
    """Chain into a Label-Studio populate child on this worker's queue.

    Also ALLOW_DUPLICATE, and for a sharper reason than the preprocess
    child: a *completed* populate burning the id permanently stalls any
    dive that later gains an eligible image — it never gets an LS task,
    never gets a label row, so it never drains from the preprocess cohort,
    blocking every higher-id dive behind it while re-staging its raw
    `.ORF`s from NAS every hour. Prod dive 60 held up dives 84/465/471
    that way until 2026-08-04.

    Dedup lives *inside* the child, where it also protects manual runs:
    the populate activity selects only images with no non-sentinel label
    row, and `import_tasks_and_record_labels` dedupes by URL against tasks
    already in the project.
    """
    try:
        await workflow.execute_child_workflow(
            workflow_name,
            dive_id,
            id=child_id,
            execution_timeout=timedelta(minutes=30),
            id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
        )
    except WorkflowAlreadyStartedError:
        # Only reachable while a prior populate for this dive is still
        # RUNNING (manual run overlapping the schedule).
        workflow.logger.info(
            "%s is still running; skipping duplicate dispatch", child_id
        )
