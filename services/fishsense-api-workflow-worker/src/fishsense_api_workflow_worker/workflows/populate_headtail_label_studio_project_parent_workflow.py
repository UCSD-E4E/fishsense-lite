"""Scheduled parent: (re)populate head/tail LS tasks for every dive needing it.

Decoupled from the stage-5.1 preprocess parent, which used to dispatch populate
as a child immediately after writing JPEGs. That ordering cannot survive a
prediction stage: populate seeds sentinel `HeadTailLabel` rows and the predict
cohort excludes any image with a live label, so populating first would starve
every image of a prediction permanently. The laser stage hit exactly this and
fixed it the same way, with a +10/+12 predict/populate pair.

Selects the prediction-gated "needs head/tail population" cohort (see the api's
`needing-headtail-population` endpoint) and fans out one
`PopulateHeadTailLabelStudioProjectWorkflow` child per dive.

Runs at +34, after the head/tail predict parent at +32 has written
`HeadTailPrediction` rows — the populate activity only seeds a task for an
image that already has one. Idempotent and prediction-gated, so SKIP-overlap
hourly firings converge and one dive's failure does not abort the fan-out.
"""

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError

from fishsense_shared import ExceptionGroupErrorLogging

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.workflows._retry_policies import (
        SDK_FAIL_FAST_RETRY_POLICY,
    )

# Bound on concurrent per-dive populate children — a handful of LS API calls
# each; keep modest so a backlog doesn't hammer hosted Label Studio.
POPULATE_CONCURRENCY = 4


@workflow.defn
class PopulateHeadTailLabelStudioProjectParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Fan out headtail populate across every dive needing it. Returns the list
    of dive_ids dispatched (empty when the cohort is empty)."""

    @workflow.run
    async def run(self) -> list[int]:
        dive_ids = await workflow.execute_activity(
            "select_dives_needing_headtail_population_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
        )
        if not dive_ids:
            return []

        sem = asyncio.Semaphore(POPULATE_CONCURRENCY)

        async def __populate(dive_id: int) -> None:
            async with sem:
                try:
                    await workflow.execute_child_workflow(
                        "PopulateHeadTailLabelStudioProjectWorkflow",
                        dive_id,
                        id=f"populate-headtail-{dive_id}",
                        execution_timeout=timedelta(minutes=30),
                        # ALLOW_DUPLICATE (not FAILED_ONLY): a scheduled re-run
                        # can populate newly-predicted images after the prior
                        # run closed; the child is idempotent (LS import dedupes,
                        # SDK upserts), so a no-op re-run is cheap.
                        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
                    )
                except WorkflowAlreadyStartedError:
                    workflow.logger.info(
                        "populate-headtail-%d already running; skipping duplicate "
                        "dispatch",
                        dive_id,
                    )

        # suppress=True: one dive's failure must not abort the fan-out.
        async with ExceptionGroupErrorLogging(workflow.logger, suppress=True):
            async with asyncio.TaskGroup() as tg:
                for dive_id in dive_ids:
                    tg.create_task(__populate(dive_id))

        return dive_ids
