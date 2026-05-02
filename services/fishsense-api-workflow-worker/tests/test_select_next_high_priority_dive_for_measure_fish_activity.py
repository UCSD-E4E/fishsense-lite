# pylint: disable=unused-argument
"""Unit tests for select_next_high_priority_dive_for_measure_fish_activity.

Pins down:
  1. Skip LOW-priority dives.
  2. Skip dives without `LaserExtrinsics` (stage 13 prerequisite).
  3. Skip dives with no LABEL_STUDIO clusters.
  4. Skip dives whose LABEL_STUDIO clusters all have `fish_id` set
     (already measured on a fully-successful previous run).
  5. Pick the lowest-id dive among the remaining cohort.
  6. Empty cohort -> None.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.data_source import DataSource
from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.dive_frame_cluster import DiveFrameCluster
from fishsense_api_sdk.models.laser_extrinsics import LaserExtrinsics
from fishsense_api_workflow_worker.activities import (
    select_next_high_priority_dive_for_measure_fish_activity as sut,
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
        dive_slate_id=7,
    )


def _extrinsics() -> LaserExtrinsics:
    return LaserExtrinsics(
        laser_position=np.zeros(3),
        laser_axis=np.array([0.0, 0.0, 1.0]),
        dive_id=1,
        camera_id=1,
    )


def _cluster(dive_id: int, *, fish_id: int | None) -> DiveFrameCluster:
    return DiveFrameCluster(
        id=None,
        image_ids=[1],
        data_source=DataSource.LABEL_STUDIO,
        updated_at=None,
        dive_id=dive_id,
        fish_id=fish_id,
    )


def _make_fs(
    *,
    dives: List[Dive],
    extrinsics_for: dict[int, LaserExtrinsics],
    clusters_for: dict[int, list[DiveFrameCluster]],
):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(return_value=dives)

    async def _get_le(dive_id: int) -> Optional[LaserExtrinsics]:
        return extrinsics_for.get(dive_id)

    fs.dives.get_laser_extrinsics = AsyncMock(side_effect=_get_le)

    fs.images = MagicMock()

    async def _get_clusters(dive_id: int, data_source: str):
        assert data_source == DataSource.LABEL_STUDIO.value
        return clusters_for.get(dive_id, [])

    fs.images.get_clusters = AsyncMock(side_effect=_get_clusters)
    return fs


@pytest.mark.asyncio
async def test_picks_lowest_id_dive_with_unbound_clusters(monkeypatch):
    fs = _make_fs(
        dives=[_dive(2), _dive(1), _dive(3, priority="LOW")],
        extrinsics_for={1: _extrinsics(), 2: _extrinsics()},
        clusters_for={
            1: [_cluster(1, fish_id=None)],
            2: [_cluster(2, fish_id=None)],
        },
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_measure_fish_activity
    )

    assert result == 1


@pytest.mark.asyncio
async def test_skips_dives_without_laser_extrinsics(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1), _dive(2)],
        extrinsics_for={2: _extrinsics()},
        clusters_for={
            1: [_cluster(1, fish_id=None)],
            2: [_cluster(2, fish_id=None)],
        },
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_measure_fish_activity
    )

    assert result == 2


@pytest.mark.asyncio
async def test_skips_dives_without_label_studio_clusters(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1), _dive(2)],
        extrinsics_for={1: _extrinsics(), 2: _extrinsics()},
        clusters_for={
            1: [],
            2: [_cluster(2, fish_id=None)],
        },
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_measure_fish_activity
    )

    assert result == 2


@pytest.mark.asyncio
async def test_skips_dives_with_all_clusters_already_bound(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1), _dive(2)],
        extrinsics_for={1: _extrinsics(), 2: _extrinsics()},
        clusters_for={
            1: [_cluster(1, fish_id=11), _cluster(1, fish_id=12)],
            2: [_cluster(2, fish_id=None)],
        },
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_measure_fish_activity
    )

    assert result == 2


@pytest.mark.asyncio
async def test_returns_none_when_no_eligible_dives(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1, priority="LOW"), _dive(2)],
        extrinsics_for={2: _extrinsics()},
        # Dive 2 has all clusters bound -> not eligible.
        clusters_for={2: [_cluster(2, fish_id=99)]},
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_measure_fish_activity
    )

    assert result is None


@pytest.mark.asyncio
async def test_returns_none_when_dives_endpoint_returns_empty(monkeypatch):
    fs = _make_fs(dives=[], extrinsics_for={}, clusters_for={})
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_measure_fish_activity
    )

    assert result is None
