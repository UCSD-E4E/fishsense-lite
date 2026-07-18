"""Scheduled parent: (re)populate species LS tasks for every dive needing it.

Selects the superseded-aware "needs species population" cohort (see the
api's `needing-species-population` endpoint) and fans out one
`PopulateSpeciesLabelStudioProjectWorkflow` child per dive. Populate is
now the decoupled, scheduled home for species task import — the stage-2
preprocess parent no longer chains into it.

Safe to run on a schedule because the populate activity is idempotent
(a dive with no new laser-valid images imports nothing) and JPEG-gated
(images whose stage-2 JPEG isn't in Garage yet are deferred to a later
run rather than seeded ahead of preprocess). One dive's failure doesn't
abort the rest of the fan-out.
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

# Bound on concurrent per-dive populate children. Each is a handful of
# LS API calls + SDK upserts; keep it modest so a large backlog doesn't
# hammer the hosted Label Studio import endpoint.
POPULATE_CONCURRENCY = 4


@workflow.defn
class PopulateSpeciesLabelStudioProjectParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Fan out species populate across every dive needing it.

    Returns the list of dive_ids it dispatched (empty when the cohort is
    empty).
    """

    @workflow.run
    async def run(self) -> list[int]:
        dive_ids = await workflow.execute_activity(
            "select_dives_needing_species_population_activity",
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
                        "PopulateSpeciesLabelStudioProjectWorkflow",
                        dive_id,
                        # Deterministic id dedupes against an overlapping
                        # firing; ALLOW_DUPLICATE (not FAILED_ONLY) so a
                        # scheduled re-run can populate newly-laser-valid
                        # images after the prior run closed — the child is
                        # idempotent, so a no-op re-run is cheap.
                        id=f"populate-species-{dive_id}",
                        execution_timeout=timedelta(minutes=30),
                        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
                    )
                except WorkflowAlreadyStartedError:
                    workflow.logger.info(
                        "populate-species-%d already running; skipping "
                        "duplicate dispatch",
                        dive_id,
                    )

        # suppress=True: one dive's populate failure must not abort the
        # whole fan-out (mirrors SyncLabelStudioLaserLabelsWorkflow's
        # validation pass).
        async with ExceptionGroupErrorLogging(workflow.logger, suppress=True):
            async with asyncio.TaskGroup() as tg:
                for dive_id in dive_ids:
                    tg.create_task(__populate(dive_id))

        return dive_ids
