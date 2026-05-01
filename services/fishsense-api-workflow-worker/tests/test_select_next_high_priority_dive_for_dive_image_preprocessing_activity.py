# pylint: disable=unused-argument
"""Unit tests for select_next_high_priority_dive_for_dive_image_preprocessing_activity.

Pins down:
  1. Skip LOW-priority dives.
  2. Skip dives without PREDICTION clusters (stage 1 hasn't run).
  3. Skip dives where every image has a completed species label.
  4. Pick the lowest-id dive in the remaining cohort.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.data_source import DataSource
from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.dive_frame_cluster import DiveFrameCluster
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.species_label import SpeciesLabel
from fishsense_api_workflow_worker.activities import (
    select_next_high_priority_dive_for_dive_image_preprocessing_activity as sut,
)


def _dive(dive_id: int, *, priority: str = "HIGH") -> Dive:
    return Dive(
        id=dive_id,
        name=f"dive-{dive_id}",
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority=priority,
        flip_dive_slate=False,
        camera_id=1,
        dive_slate_id=None,
    )


def _image(image_id: int) -> Image:
    return Image(
        id=image_id,
        path=f"/dev/null/{image_id}",
        taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        checksum=f"chk-{image_id}",
        is_canonical=True,
        dive_id=42,
        camera_id=1,
    )


def _cluster(cluster_id: int, image_ids: List[int]) -> DiveFrameCluster:
    return DiveFrameCluster(
        id=cluster_id,
        image_ids=image_ids,
        data_source=DataSource.PREDICTION,
        updated_at=None,
        dive_id=42,
        fish_id=None,
    )


def _species(image_id: int, *, completed: bool) -> SpeciesLabel:
    return SpeciesLabel(
        id=None,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=70,
        image_url=None,
        updated_at=None,
        completed=completed,
        grouping=None,
        top_three_photos_of_group=None,
        slate_upside_down=None,
        laser_x=None,
        laser_y=None,
        laser_label=None,
        content_of_image=None,
        fish_measurable_category=None,
        fish_angle_category=None,
        fish_curved_category=None,
        label_studio_json={},
        image_id=image_id,
        user_id=None,
    )


def _make_fs(
    *,
    dives: List[Dive],
    clusters_by_dive: dict[int, List[DiveFrameCluster]],
    images_by_dive: dict[int, List[Image]],
    species_by_dive: dict[int, List[SpeciesLabel]],
):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(return_value=dives)

    fs.images = MagicMock()

    async def _get_clusters(dive_id, data_source):
        if data_source != DataSource.PREDICTION.value:
            return []
        return clusters_by_dive.get(dive_id, [])

    fs.images.get_clusters = AsyncMock(side_effect=_get_clusters)

    async def _get_images(dive_id=None, **_):
        return images_by_dive.get(dive_id, [])

    fs.images.get = AsyncMock(side_effect=_get_images)

    fs.labels = MagicMock()

    async def _get_species(dive_id):
        return species_by_dive.get(dive_id, [])

    fs.labels.get_species_labels = AsyncMock(side_effect=_get_species)
    return fs


@pytest.mark.asyncio
async def test_picks_lowest_id_high_dive_with_clusters_and_incomplete_species(
    monkeypatch,
):
    fs = _make_fs(
        dives=[_dive(2), _dive(1), _dive(3, priority="LOW")],
        clusters_by_dive={
            1: [_cluster(10, [101, 102])],
            2: [_cluster(20, [201])],
        },
        images_by_dive={
            1: [_image(101), _image(102)],
            2: [_image(201)],
        },
        species_by_dive={
            1: [_species(101, completed=False)],
            2: [_species(201, completed=True)],
        },
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_dive_image_preprocessing_activity
    )
    assert result == 1


@pytest.mark.asyncio
async def test_skips_dive_without_prediction_clusters(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1), _dive(2)],
        clusters_by_dive={2: [_cluster(20, [201])]},
        images_by_dive={1: [_image(101)], 2: [_image(201)]},
        species_by_dive={
            1: [_species(101, completed=False)],
            2: [_species(201, completed=False)],
        },
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_dive_image_preprocessing_activity
    )
    assert result == 2


@pytest.mark.asyncio
async def test_skips_dive_when_all_species_labels_completed(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1)],
        clusters_by_dive={1: [_cluster(10, [101])]},
        images_by_dive={1: [_image(101)]},
        species_by_dive={1: [_species(101, completed=True)]},
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_dive_image_preprocessing_activity
    )
    assert result is None


@pytest.mark.asyncio
async def test_picks_dive_when_image_has_no_species_label(monkeypatch):
    # Image without any species label row -> counts as incomplete.
    fs = _make_fs(
        dives=[_dive(1)],
        clusters_by_dive={1: [_cluster(10, [101])]},
        images_by_dive={1: [_image(101)]},
        species_by_dive={1: []},
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_dive_image_preprocessing_activity
    )
    assert result == 1


@pytest.mark.asyncio
async def test_returns_none_for_empty_dives_endpoint(monkeypatch):
    fs = _make_fs(
        dives=[],
        clusters_by_dive={},
        images_by_dive={},
        species_by_dive={},
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_dive_image_preprocessing_activity
    )
    assert result is None
