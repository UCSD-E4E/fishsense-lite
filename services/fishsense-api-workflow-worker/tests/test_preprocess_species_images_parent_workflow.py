# pylint: disable=unused-argument
"""Workflow contract test for PreprocessSpeciesImagesParentWorkflow."""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import List, Optional

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows.preprocess_species_images_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    DATA_PROCESSING_TASK_QUEUE,
    PreprocessSpeciesImagesParentWorkflow,
)
from fishsense_shared import PreprocessSpeciesImagesInput


_K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
_D = [-0.1, 0.05, 0.0, 0.0, 0.0]


@workflow.defn(name="PreprocessSpeciesImagesWorkflow")
class _StubChildWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, payload: PreprocessSpeciesImagesInput) -> None:
        flat = [c for cluster in payload.clusters for c in cluster]
        await workflow.execute_activity(
            "_record_child_dispatch",
            args=(workflow.info().workflow_id, payload.dive_id, flat),
            schedule_to_close_timeout=timedelta(seconds=5),
        )


@workflow.defn(name="PopulateSpeciesLabelStudioProjectWorkflow")
class _StubPopulateWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, dive_id: int) -> int:
        await workflow.execute_activity(
            "_record_populate_dispatch",
            args=(workflow.info().workflow_id, dive_id),
            schedule_to_close_timeout=timedelta(seconds=5),
        )
        return 0


def _make_recording_activity(captures: List[tuple]):
    @activity.defn(name="_record_child_dispatch")
    async def record_child_dispatch(
        workflow_id: str, dive_id: int, checksums: List[str]
    ) -> None:
        captures.append((workflow_id, dive_id, checksums))

    return record_child_dispatch


def _make_populate_recording_activity(captures: List[tuple]):
    @activity.defn(name="_record_populate_dispatch")
    async def record_populate_dispatch(workflow_id: str, dive_id: int) -> None:
        captures.append((workflow_id, dive_id))

    return record_populate_dispatch


def _make_stubs(
    selector_result: Optional[int],
    resolver_result: Optional[PreprocessSpeciesImagesInput],
):
    selector_calls: List[None] = []
    resolver_calls: List[int] = []

    @activity.defn(name="select_next_high_priority_dive_for_species_preprocessing_activity")
    async def stub_select() -> Optional[int]:
        selector_calls.append(None)
        return selector_result

    @activity.defn(name="resolve_species_preprocess_inputs_activity")
    async def stub_resolve(dive_id: int) -> PreprocessSpeciesImagesInput:
        resolver_calls.append(dive_id)
        assert resolver_result is not None
        return resolver_result

    @activity.defn(name="stage_raw_bytes_for_dive_activity")
    async def stub_stage(dive_id: int) -> None:
        return None

    @activity.defn(name="cleanup_raw_bytes_for_dive_activity")
    async def stub_cleanup(dive_id: int) -> None:
        return None

    @activity.defn(name="ensure_data_worker_running_activity")
    async def stub_ensure_running() -> int:
        return 0

    return (
        [
            stub_select,
            stub_resolve,
            stub_stage,
            stub_cleanup,
            stub_ensure_running,
        ],
        selector_calls,
        resolver_calls,
    )


@pytest.mark.asyncio
async def test_dispatches_child_with_deterministic_id_and_clusters():
    inputs = PreprocessSpeciesImagesInput(
        dive_id=440,
        clusters=[["a", "b"], ["c"]],
        camera_matrix=_K,
        distortion_coefficients=_D,
    )
    activities, selector_calls, resolver_calls = _make_stubs(440, inputs)
    child_runs: List[tuple] = []
    populate_runs: List[tuple] = []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage2-parent",
            workflows=[
                PreprocessSpeciesImagesParentWorkflow,
                _StubPopulateWorkflow,
            ],
            activities=[
                *activities,
                _make_populate_recording_activity(populate_runs),
            ],
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[_make_recording_activity(child_runs)],
        ):
            result = await env.client.execute_workflow(
                PreprocessSpeciesImagesParentWorkflow.run,
                id=f"test-stage2-parent-{uuid.uuid4()}",
                task_queue="test-stage2-parent",
            )

    assert result == 440
    assert len(selector_calls) == 1
    assert resolver_calls == [440]
    assert len(child_runs) == 1
    child_id, child_dive_id, flat = child_runs[0]
    assert child_id == "preprocess-species-440"
    assert child_dive_id == 440
    assert flat == ["a", "b", "c"]
    # Populate is decoupled now — the preprocess parent no longer chains
    # into PopulateSpeciesLabelStudioProjectWorkflow (the scheduled
    # PopulateSpeciesLabelStudioProjectParentWorkflow owns it).
    assert not populate_runs


@pytest.mark.asyncio
async def test_returns_none_when_selector_finds_no_dive():
    activities, _, resolver_calls = _make_stubs(None, None)
    child_runs: List[tuple] = []
    populate_runs: List[tuple] = []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage2-parent-empty",
            workflows=[
                PreprocessSpeciesImagesParentWorkflow,
                _StubPopulateWorkflow,
            ],
            activities=[
                *activities,
                _make_populate_recording_activity(populate_runs),
            ],
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[_make_recording_activity(child_runs)],
        ):
            result = await env.client.execute_workflow(
                PreprocessSpeciesImagesParentWorkflow.run,
                id=f"test-stage2-parent-empty-{uuid.uuid4()}",
                task_queue="test-stage2-parent-empty",
            )

    assert result is None
    assert not resolver_calls
    assert not child_runs
    assert not populate_runs


@pytest.mark.asyncio
async def test_skips_child_dispatch_when_no_clusters():
    inputs = PreprocessSpeciesImagesInput(
        dive_id=440,
        clusters=[],
        camera_matrix=_K,
        distortion_coefficients=_D,
    )
    activities, _, _ = _make_stubs(440, inputs)
    child_runs: List[tuple] = []
    populate_runs: List[tuple] = []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage2-parent-empty-clusters",
            workflows=[
                PreprocessSpeciesImagesParentWorkflow,
                _StubPopulateWorkflow,
            ],
            activities=[
                *activities,
                _make_populate_recording_activity(populate_runs),
            ],
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[_make_recording_activity(child_runs)],
        ):
            result = await env.client.execute_workflow(
                PreprocessSpeciesImagesParentWorkflow.run,
                id=f"test-stage2-parent-empty-clusters-{uuid.uuid4()}",
                task_queue="test-stage2-parent-empty-clusters",
            )

    assert result == 440
    assert not child_runs
    assert not populate_runs


@pytest.mark.asyncio
async def test_child_redispatches_after_a_prior_successful_run():
    """The preprocess child must re-run on a dive it already processed.

    A dive's image set grows after its first successful child run — a laser
    validated after one-shot stage-1 clustering, or an orphan later given a
    cluster. Under ALLOW_DUPLICATE_FAILED_ONLY the deterministic child id
    (`preprocess-species-{dive}`) was permanently spent once it completed, so
    those new images' JPEGs were never produced and populate deferred them
    forever. With ALLOW_DUPLICATE the child re-dispatches; the resolver
    returns only still-needed images and per-image work is idempotent, so
    re-running is safe.

    Two parent runs on the same dive → the child (same deterministic id) must
    run BOTH times.
    """
    inputs = PreprocessSpeciesImagesInput(
        dive_id=59,
        clusters=[["orphan-checksum"]],
        camera_matrix=_K,
        distortion_coefficients=_D,
    )
    activities, _, _ = _make_stubs(59, inputs)
    child_runs: List[tuple] = []
    populate_runs: List[tuple] = []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage2-redispatch",
            workflows=[
                PreprocessSpeciesImagesParentWorkflow,
                _StubPopulateWorkflow,
            ],
            activities=[
                *activities,
                _make_populate_recording_activity(populate_runs),
            ],
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[_make_recording_activity(child_runs)],
        ):
            for _ in range(2):
                await env.client.execute_workflow(
                    PreprocessSpeciesImagesParentWorkflow.run,
                    id=f"test-stage2-redispatch-parent-{uuid.uuid4()}",
                    task_queue="test-stage2-redispatch",
                )

    # Same deterministic child id both times; both must have run.
    assert len(child_runs) == 2, "child must re-dispatch on the second parent run"
    assert {c[0] for c in child_runs} == {"preprocess-species-59"}
