"""Unit tests for the rewritten cluster_dive_frames activity.

The activity now consumes `(image_id, taken_datetime)` pairs rather
than the data-worker's local pydantic Image type, so the workflow-
level contract crosses the worker boundary as a small shared DTO
(`fishsense_shared.ClusterDiveFramesInput`). Output shape flips from
`list[list[Image]]` to `list[list[int]]` of image_ids — the api-worker
parent persists those id lists via `images.post_cluster`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_data_processing_workflow_worker.activities.cluster_dive_frames import (
    cluster_dive_frames,
)
from fishsense_shared import ClusterDiveFrameImage


def _img(image_id: int, ts: datetime) -> ClusterDiveFrameImage:
    return ClusterDiveFrameImage(image_id=image_id, taken_datetime=ts)


@pytest.mark.asyncio
async def test_returns_image_ids_grouped_by_temporal_proximity():
    """Two well-separated dense clusters (5 images each, 10 minutes
    apart) reliably resolve into two HDBSCAN clusters with default
    parameters. The contract check: output is `list[list[int]]` of
    image_ids, every input id appears, two distinct clusters survive."""
    base = datetime(2026, 5, 5, 10, 0, 0, tzinfo=timezone.utc)
    cluster_a = [_img(i, base + timedelta(seconds=i)) for i in range(1, 6)]
    cluster_b = [
        _img(10 + i, base + timedelta(minutes=10, seconds=i)) for i in range(1, 6)
    ]
    images = cluster_a + cluster_b

    result = await ActivityEnvironment().run(cluster_dive_frames, images)

    assert isinstance(result, list)
    assert all(isinstance(c, list) for c in result)
    assert all(isinstance(image_id, int) for c in result for image_id in c)
    flat = sorted(image_id for c in result for image_id in c)
    assert flat == sorted(img.image_id for img in images)
    assert len(result) == 2
    cluster_sets = [set(c) for c in result]
    assert {img.image_id for img in cluster_a} in cluster_sets
    assert {img.image_id for img in cluster_b} in cluster_sets


@pytest.mark.asyncio
async def test_returns_empty_when_no_images():
    result = await ActivityEnvironment().run(cluster_dive_frames, [])
    assert result == []
