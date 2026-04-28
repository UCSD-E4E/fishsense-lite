"""Workflow contract test for PreprocessDiveImagesWorkflow.

Runs the workflow end-to-end against an in-process Temporal test server
(`WorkflowEnvironment.start_time_skipping()` — no real temporald). The
preprocess_dive_image activity is replaced with a stub that records its
inputs so we can assert the workflow's fanout and per-image arg shape
without doing any real image work.
"""

import asyncio
import uuid
from typing import List

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.workflows.preprocess_dive_images_workflow import (
    PreprocessDiveImageInput,
    PreprocessDiveImagesInput,
    PreprocessDiveImagesWorkflow,
)


# A small but realistic intrinsics shape (3x3 matrix, 5-element distortion
# vector). Values are arbitrary — the stub activity doesn't use them.
_K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
_D = [-0.1, 0.05, 0.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_workflow_fans_out_one_activity_per_image_with_correct_indices():
    calls: List[PreprocessDiveImageInput] = []

    @activity.defn(name="preprocess_dive_image")
    async def stub_preprocess_dive_image(
        input: PreprocessDiveImageInput,
    ) -> None:
        calls.append(input)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage2",
            workflows=[PreprocessDiveImagesWorkflow],
            activities=[stub_preprocess_dive_image],
        ):
            await env.client.execute_workflow(
                PreprocessDiveImagesWorkflow.run,
                PreprocessDiveImagesInput(
                    dive_id=383,
                    clusters=[
                        ["cs1", "cs2", "cs3"],
                        ["cs4", "cs5"],
                    ],
                    camera_matrix=_K,
                    distortion_coefficients=_D,
                ),
                id=f"test-stage2-{uuid.uuid4()}",
                task_queue="test-stage2",
            )

    assert len(calls) == 5
    by_checksum = {c.checksum: c for c in calls}

    # cluster 1: 3 images, 1-based indices 1..3, size 3
    assert (by_checksum["cs1"].cluster_index, by_checksum["cs1"].cluster_size) == (1, 3)
    assert (by_checksum["cs2"].cluster_index, by_checksum["cs2"].cluster_size) == (2, 3)
    assert (by_checksum["cs3"].cluster_index, by_checksum["cs3"].cluster_size) == (3, 3)
    # cluster 2: 2 images, 1-based indices 1..2, size 2
    assert (by_checksum["cs4"].cluster_index, by_checksum["cs4"].cluster_size) == (1, 2)
    assert (by_checksum["cs5"].cluster_index, by_checksum["cs5"].cluster_size) == (2, 2)

    # Output folder is fixed for stage2 — the JPEGs land alongside the
    # existing labeler-facing /api/v1/data/preprocess_groups_jpeg/ route.
    assert {c.output_folder for c in calls} == {"preprocess_groups_jpeg"}

    # Camera intrinsics are propagated unchanged to every image.
    for c in calls:
        assert c.camera_matrix == _K
        assert c.distortion_coefficients == _D


@pytest.mark.asyncio
async def test_workflow_with_no_clusters_makes_no_activity_calls():
    calls: List[PreprocessDiveImageInput] = []

    @activity.defn(name="preprocess_dive_image")
    async def stub_preprocess_dive_image(
        input: PreprocessDiveImageInput,
    ) -> None:
        calls.append(input)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage2-empty",
            workflows=[PreprocessDiveImagesWorkflow],
            activities=[stub_preprocess_dive_image],
        ):
            await env.client.execute_workflow(
                PreprocessDiveImagesWorkflow.run,
                PreprocessDiveImagesInput(
                    dive_id=383,
                    clusters=[],
                    camera_matrix=_K,
                    distortion_coefficients=_D,
                ),
                id=f"test-stage2-empty-{uuid.uuid4()}",
                task_queue="test-stage2-empty",
            )

    assert calls == []
