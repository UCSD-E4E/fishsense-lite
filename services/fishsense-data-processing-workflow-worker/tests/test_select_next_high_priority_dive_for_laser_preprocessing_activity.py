"""Unit tests for select_next_high_priority_dive_for_laser_preprocessing_activity.

Pins down four things:
  1. Skip LOW-priority dives.
  2. Skip HIGH-priority dives that already have laser_extrinsics.
  3. Pick the lowest-id dive among the remaining cohort.
  4. Empty cohort -> None.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.laser_extrinsics import LaserExtrinsics
from fishsense_data_processing_workflow_worker.activities import (
    select_next_high_priority_dive_for_laser_preprocessing_activity as sut,
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


def _extrinsics() -> LaserExtrinsics:
    return LaserExtrinsics(
        laser_position=np.zeros(3),
        laser_axis=np.array([0.0, 0.0, 1.0]),
        dive_id=1,
        camera_id=1,
    )


def _make_fs(*, dives: List[Dive], extrinsics_for: dict[int, LaserExtrinsics]):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(return_value=dives)

    async def _get_le(dive_id: int) -> Optional[LaserExtrinsics]:
        return extrinsics_for.get(dive_id)

    fs.dives.get_laser_extrinsics = AsyncMock(side_effect=_get_le)
    return fs


@pytest.mark.asyncio
async def test_picks_lowest_id_high_priority_dive_without_extrinsics(monkeypatch):
    fs = _make_fs(
        dives=[
            _dive(2, priority="HIGH"),
            _dive(1, priority="HIGH"),
            _dive(3, priority="LOW"),
        ],
        extrinsics_for={},
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_laser_preprocessing_activity
    )

    assert result == 1


@pytest.mark.asyncio
async def test_skips_high_priority_dives_with_extrinsics(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1), _dive(2), _dive(3)],
        extrinsics_for={1: _extrinsics(), 2: _extrinsics()},
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_laser_preprocessing_activity
    )

    assert result == 3


@pytest.mark.asyncio
async def test_returns_none_when_no_low_priority_or_uncalibrated_dives(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1, priority="LOW"), _dive(2, priority="LOW")],
        extrinsics_for={},
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_laser_preprocessing_activity
    )

    assert result is None


@pytest.mark.asyncio
async def test_returns_none_when_all_high_priority_dives_have_extrinsics(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1), _dive(2)],
        extrinsics_for={1: _extrinsics(), 2: _extrinsics()},
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_laser_preprocessing_activity
    )

    assert result is None


@pytest.mark.asyncio
async def test_returns_none_when_dives_endpoint_returns_empty(monkeypatch):
    fs = _make_fs(dives=[], extrinsics_for={})
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_laser_preprocessing_activity
    )

    assert result is None
