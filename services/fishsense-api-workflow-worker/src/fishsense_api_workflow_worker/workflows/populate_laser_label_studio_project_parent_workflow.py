"""Scheduled parent: (re)populate laser LS tasks for every dive needing it.

Decoupled from the stage-0.1 preprocess parent (2026-07-28, model-assisted
labeling). Selects the prediction-gated "needs laser population" cohort (see
the api's `needing-laser-population` endpoint) and fans out one
`PopulateLaserLabelStudioProjectWorkflow` child per dive.

Runs at +12 min, after the laser-detector predict parent (+10) has written
`LaserPrediction` rows — the populate activity only seeds a task (and its
non-sentinel `LaserLabel` row) for an image that already has a prediction, so
it never starves the predict cohort. Idempotent + prediction-gated, so
SKIP-overlap hourly firings converge and one dive's failure doesn't abort the
fan-out.
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
class PopulateLaserLabelStudioProjectParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Fan out laser populate across every dive needing it. Returns the list
    of dive_ids dispatched (empty when the cohort is empty)."""

    @workflow.run
    async def run(self) -> list[int]:
        dive_ids = await workflow.execute_activity(
            "select_dives_needing_laser_population_activity",
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
                        "PopulateLaserLabelStudioProjectWorkflow",
                        dive_id,
                        id=f"populate-laser-{dive_id}",
                        execution_timeout=timedelta(minutes=30),
                        # ALLOW_DUPLICATE (not FAILED_ONLY): a scheduled re-run
                        # can populate newly-predicted images after the prior
                        # run closed; the child is idempotent (LS import dedupes,
                        # SDK upserts), so a no-op re-run is cheap.
                        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
                    )
                except WorkflowAlreadyStartedError:
                    workflow.logger.info(
                        "populate-laser-%d already running; skipping duplicate "
                        "dispatch",
                        dive_id,
                    )

        # suppress=True: one dive's failure must not abort the fan-out.
        async with ExceptionGroupErrorLogging(workflow.logger, suppress=True):
            async with asyncio.TaskGroup() as tg:
                for dive_id in dive_ids:
                    tg.create_task(__populate(dive_id))

        return dive_ids
