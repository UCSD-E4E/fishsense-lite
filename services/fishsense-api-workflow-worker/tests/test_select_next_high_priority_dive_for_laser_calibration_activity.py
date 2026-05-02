# pylint: disable=unused-argument
"""Unit tests for select_next_high_priority_dive_for_laser_calibration_activity.

Pins down:
  1. Skip LOW-priority dives.
  2. Skip dives without a `dive_slate_id`.
  3. Skip dives that already have laser_extrinsics.
  4. Skip dives with fewer than `MIN_COMPLETED_SLATE_LABELS` (=2) completed
     slate labels — under that floor the data-worker activity raises
     `ValueError`, and the schedule would otherwise fire forever.
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

from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
from fishsense_api_sdk.models.laser_extrinsics import LaserExtrinsics
from fishsense_api_workflow_worker.activities import (
    select_next_high_priority_dive_for_laser_calibration_activity as sut,
)


def _dive(
    dive_id: int, *, priority: str = "HIGH", dive_slate_id: int | None = 7
) -> Dive:
    return Dive(
        id=dive_id,
        name=f"dive-{dive_id}",
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority=priority,
        flip_dive_slate=False,
        camera_id=1,
        dive_slate_id=dive_slate_id,
    )


def _extrinsics() -> LaserExtrinsics:
    return LaserExtrinsics(
        laser_position=np.zeros(3),
        laser_axis=np.array([0.0, 0.0, 1.0]),
        dive_id=1,
        camera_id=1,
    )


def _slate_label(image_id: int, *, completed: bool) -> DiveSlateLabel:
    return DiveSlateLabel(
        id=None,
        label_studio_task_id=None,
        label_studio_project_id=None,
        image_url=None,
        upside_down=False,
        reference_points=[(0.0, 0.0)],
        slate_rectangle=None,
        skipped_points=[],
        updated_at=None,
        completed=completed,
        label_studio_json=None,
        image_id=image_id,
        user_id=None,
    )


def _make_fs(
    *,
    dives: List[Dive],
    extrinsics_for: dict[int, LaserExtrinsics],
    slate_labels_for: dict[int, list[DiveSlateLabel]],
):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(return_value=dives)

    async def _get_le(dive_id: int) -> Optional[LaserExtrinsics]:
        return extrinsics_for.get(dive_id)

    fs.dives.get_laser_extrinsics = AsyncMock(side_effect=_get_le)

    fs.labels = MagicMock()

    async def _get_slate(dive_id: int) -> list[DiveSlateLabel]:
        return slate_labels_for.get(dive_id, [])

    fs.labels.get_dive_slate_labels = AsyncMock(side_effect=_get_slate)
    return fs


@pytest.mark.asyncio
async def test_picks_lowest_id_high_priority_dive_meeting_cohort(monkeypatch):
    fs = _make_fs(
        dives=[
            _dive(2, priority="HIGH"),
            _dive(1, priority="HIGH"),
            _dive(3, priority="LOW"),
        ],
        extrinsics_for={},
        slate_labels_for={
            1: [
                _slate_label(101, completed=True),
                _slate_label(102, completed=True),
            ],
            2: [
                _slate_label(201, completed=True),
                _slate_label(202, completed=True),
            ],
        },
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_laser_calibration_activity
    )

    assert result == 1


@pytest.mark.asyncio
async def test_skips_dives_without_dive_slate_id(monkeypatch):
    fs = _make_fs(
        dives=[
            _dive(1, dive_slate_id=None),
            _dive(2, dive_slate_id=7),
        ],
        extrinsics_for={},
        slate_labels_for={
            2: [
                _slate_label(201, completed=True),
                _slate_label(202, completed=True),
            ],
        },
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_laser_calibration_activity
    )

    assert result == 2


@pytest.mark.asyncio
async def test_skips_dives_already_calibrated(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1), _dive(2), _dive(3)],
        extrinsics_for={1: _extrinsics(), 2: _extrinsics()},
        slate_labels_for={
            3: [
                _slate_label(301, completed=True),
                _slate_label(302, completed=True),
            ],
        },
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_laser_calibration_activity
    )

    assert result == 3


@pytest.mark.asyncio
async def test_skips_dives_below_completed_slate_label_floor(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1), _dive(2)],
        extrinsics_for={},
        slate_labels_for={
            # Only 1 completed -> below the 2-label floor.
            1: [
                _slate_label(101, completed=True),
                _slate_label(102, completed=False),
            ],
            # 2 completed -> selectable.
            2: [
                _slate_label(201, completed=True),
                _slate_label(202, completed=True),
            ],
        },
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_laser_calibration_activity
    )

    assert result == 2


@pytest.mark.asyncio
async def test_returns_none_when_no_eligible_dives(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1, priority="LOW"), _dive(2, dive_slate_id=None)],
        extrinsics_for={},
        slate_labels_for={},
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_laser_calibration_activity
    )

    assert result is None


@pytest.mark.asyncio
async def test_returns_none_when_dives_endpoint_returns_empty(monkeypatch):
    fs = _make_fs(dives=[], extrinsics_for={}, slate_labels_for={})
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_laser_calibration_activity
    )

    assert result is None
