"""Workflow contract test for the rewritten DiveFrameClusteringWorkflow.

The workflow is a thin wrapper that delegates to the
`cluster_dive_frames` activity. After the rewrite it consumes
`ClusterDiveFramesInput` (one shared DTO across the worker boundary)
and returns `list[list[int]]`.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterable, List

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.workflows.dive_frame_clustering_workflow import (  # noqa: E501  pylint: disable=line-too-long
    DiveFrameClusteringWorkflow,
)
from fishsense_shared import ClusterDiveFrameImage, ClusterDiveFramesInput


@pytest.mark.asyncio
async def test_workflow_passes_images_to_activity_and_returns_result():
    captured: List[List[ClusterDiveFrameImage]] = []

    @activity.defn(name="cluster_dive_frames")
    async def stub_cluster(
        images: Iterable[ClusterDiveFrameImage],
    ) -> List[List[int]]:
        materialized = list(images)
        captured.append(materialized)
        return [[img.image_id for img in materialized]]

    base = datetime(2026, 5, 5, 10, 0, 0, tzinfo=timezone.utc)
    payload = ClusterDiveFramesInput(
        dive_id=440,
        images=[
            ClusterDiveFrameImage(image_id=1, taken_datetime=base),
            ClusterDiveFrameImage(
                image_id=2, taken_datetime=base + timedelta(seconds=1)
            ),
        ],
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage1-clustering",
            workflows=[DiveFrameClusteringWorkflow],
            activities=[stub_cluster],
        ):
            result = await env.client.execute_workflow(
                DiveFrameClusteringWorkflow.run,
                payload,
                id=f"test-stage1-clustering-{uuid.uuid4()}",
                task_queue="test-stage1-clustering",
            )

    assert result == [[1, 2]]
    assert len(captured) == 1
    assert [img.image_id for img in captured[0]] == [1, 2]
