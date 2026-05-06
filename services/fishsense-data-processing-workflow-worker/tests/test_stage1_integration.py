"""End-to-end stage 1 (clustering) integration test against the local
devcontainer Temporal cluster.

Stage 1 is pure math (no NAS, no file-exchange, no image bytes), so
the integration surface is just "does the data-worker workflow + the
shared input DTO actually run on a real Temporal server with a real
worker registration?" The contract test
(`test_dive_frame_clustering_workflow.py`) covers shape; this covers
deployability.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import List

import pytest
from temporalio.client import Client
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.activities.cluster_dive_frames import (
    cluster_dive_frames,
)
from fishsense_data_processing_workflow_worker.workflows.dive_frame_clustering_workflow import (  # noqa: E501  pylint: disable=line-too-long
    DiveFrameClusteringWorkflow,
)
from fishsense_shared import ClusterDiveFrameImage, ClusterDiveFramesInput


pytestmark = pytest.mark.integration


def _temporal_target() -> str:
    host = os.environ.get("FISHSENSE_TEMPORAL_HOST", "temporal")
    port = os.environ.get("FISHSENSE_TEMPORAL_PORT", "7233")
    return f"{host}:{port}"


@pytest.fixture
def configure_worker_settings(monkeypatch: pytest.MonkeyPatch):
    """Stage 1 doesn't use the file-exchange or fishsense-api, but the
    Dynaconf eager-validation guards still demand placeholders. Same
    pattern as stage 0.1 / stage 2 integration tests."""
    monkeypatch.setenv("E4EFS_FILE_EXCHANGE__URL", "http://static_file_server")
    monkeypatch.setenv("E4EFS_TEMPORAL__HOST", "temporal")
    monkeypatch.setenv("E4EFS_FISHSENSE_API__URL", "http://fishsense-api.invalid")
    yield


@pytest.mark.asyncio
@pytest.mark.usefixtures("configure_worker_settings")
async def test_workflow_clusters_a_dive_end_to_end():
    """Two well-separated cohorts (5 images each, 10 minutes apart)
    reliably resolve into two HDBSCAN clusters with default
    parameters. Returned shape is `list[list[int]]` of image_ids per
    cluster — what the api-worker's persist activity then turns into
    PREDICTION DiveFrameCluster rows."""
    base = datetime(2026, 5, 5, 10, 0, 0, tzinfo=timezone.utc)
    cohort_a = [
        ClusterDiveFrameImage(image_id=i, taken_datetime=base + timedelta(seconds=i))
        for i in range(1, 6)
    ]
    cohort_b = [
        ClusterDiveFrameImage(
            image_id=10 + i, taken_datetime=base + timedelta(minutes=10, seconds=i)
        )
        for i in range(1, 6)
    ]
    payload = ClusterDiveFramesInput(
        dive_id=-1, images=cohort_a + cohort_b
    )

    client = await Client.connect(_temporal_target())
    task_queue = f"stage1-itest-{uuid.uuid4().hex}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[DiveFrameClusteringWorkflow],
        activities=[cluster_dive_frames],
    ):
        result: List[List[int]] = await client.execute_workflow(
            DiveFrameClusteringWorkflow.run,
            payload,
            id=f"stage1-itest-{uuid.uuid4().hex}",
            task_queue=task_queue,
        )

    assert isinstance(result, list)
    assert all(isinstance(c, list) for c in result)
    assert all(isinstance(image_id, int) for c in result for image_id in c)

    flat = sorted(image_id for c in result for image_id in c)
    assert flat == sorted(img.image_id for img in payload.images)
    assert len(result) == 2
    cluster_sets = [set(c) for c in result]
    assert {img.image_id for img in cohort_a} in cluster_sets
    assert {img.image_id for img in cohort_b} in cluster_sets


@pytest.mark.asyncio
@pytest.mark.usefixtures("configure_worker_settings")
async def test_workflow_returns_empty_list_on_empty_input():
    """Edge: empty image list. The activity short-circuits before
    HDBSCAN; verify the shape contract holds across the whole
    workflow."""
    payload = ClusterDiveFramesInput(dive_id=-1, images=[])

    client = await Client.connect(_temporal_target())
    task_queue = f"stage1-itest-empty-{uuid.uuid4().hex}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[DiveFrameClusteringWorkflow],
        activities=[cluster_dive_frames],
    ):
        result = await client.execute_workflow(
            DiveFrameClusteringWorkflow.run,
            payload,
            id=f"stage1-itest-empty-{uuid.uuid4().hex}",
            task_queue=task_queue,
        )

    assert not result
