"""Unit tests for perform_laser_calibration_activity (stage 13).

End-to-end synthetic-scene test pins down the math + SDK plumbing on a
known laser line. Tolerance matches the prod-comparison thresholds in
`scripts/validate_stage13_refactor.py` (axis < 0.5 deg, position < 1 mm).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.dive_slate import DiveSlate
from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
from fishsense_api_sdk.models.laser_extrinsics import LaserExtrinsics
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_data_processing_workflow_worker.activities import (
    perform_laser_calibration_activity as sut,
)


CAMERA_MATRIX = np.array(
    [
        [3000.0, 0.0, 2048.0],
        [0.0, 3000.0, 1536.0],
        [0.0, 0.0, 1.0],
    ]
)


def _camera_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_matrix=CAMERA_MATRIX,
        distortion_coefficients=np.zeros(5),
        camera_id=1,
    )


def _slate(slate_id: int = 7) -> DiveSlate:
    pts = [
        (0.0, 0.0),
        (2400.0, 0.0),
        (0.0, 3000.0),
        (2400.0, 3000.0),
        (1200.0, 0.0),
        (1200.0, 3000.0),
    ]
    return DiveSlate(
        id=slate_id,
        name="test-slate",
        dpi=300,
        path="/dev/null",
        created_at=None,
        reference_points=pts,
    )


def _project(point_camera: np.ndarray) -> tuple[float, float]:
    p = CAMERA_MATRIX @ point_camera
    return float(p[0] / p[2]), float(p[1] / p[2])


def _dive(dive_id: int = 42, dive_slate_id: int | None = 7) -> Dive:
    return Dive(
        id=dive_id,
        name=f"dive-{dive_id}",
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority="HIGH",
        flip_dive_slate=False,
        camera_id=1,
        dive_slate_id=dive_slate_id,
    )


def _make_fs(
    dive: Dive,
    slates: List[DiveSlate],
    dive_slate_labels: List[DiveSlateLabel],
    laser_labels_by_image_id: dict[int, LaserLabel | None],
    intrinsics: CameraIntrinsics | None,
):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(return_value=dive)
    fs.dives.put_laser_extrinsics = AsyncMock(return_value=999)

    fs.dive_slates = MagicMock()
    fs.dive_slates.get = AsyncMock(return_value=slates)

    fs.cameras = MagicMock()
    fs.cameras.get_intrinsics = AsyncMock(return_value=intrinsics)

    fs.labels = MagicMock()
    fs.labels.get_dive_slate_labels = AsyncMock(return_value=dive_slate_labels)

    async def _get_laser_label(image_id: int | None = None, **_: object):
        return laser_labels_by_image_id.get(image_id)

    fs.labels.get_laser_label = AsyncMock(side_effect=_get_laser_label)

    return fs


def _build_synthetic_scene(
    n_observations: int,
    laser_origin_world: np.ndarray,
    laser_axis_world: np.ndarray,
    slate_distances: list[float],
):
    """Render n synthetic dive_slate_label + laser_label pairs.

    Each observation places the slate at a different depth along +Z
    with a small lateral offset so the PnP poses differ. The laser's
    intersection with that plane is projected to give the laser pixel.
    """
    slate = _slate()
    intrinsics = _camera_intrinsics()
    src_pts = np.array(slate.reference_points)

    body = np.zeros((len(src_pts), 3), dtype=np.float64)
    body[:, :2] = (src_pts / float(slate.dpi)) * sut.INCH_TO_M

    dive_slate_labels: list[DiveSlateLabel] = []
    laser_labels_by_image_id: dict[int, LaserLabel] = {}

    for i in range(n_observations):
        depth = slate_distances[i]
        camera_space = body.copy()
        camera_space[:, 2] = depth
        centroid_xy = body[:, :2].mean(axis=0)
        camera_space[:, 0] -= centroid_xy[0]
        camera_space[:, 1] -= centroid_xy[1]
        camera_space[:, 0] += 0.02 * (i - n_observations / 2)

        ref_pixels = [_project(p) for p in camera_space]

        # Plane normal +Z, plane offset = depth.
        t = (depth - laser_origin_world[2]) / laser_axis_world[2]
        laser_world = laser_origin_world + t * laser_axis_world
        laser_pixel = _project(laser_world)

        image_id = 100 + i
        dive_slate_labels.append(
            DiveSlateLabel(
                id=200 + i,
                label_studio_task_id=300 + i,
                label_studio_project_id=66,
                image_url=None,
                upside_down=False,
                reference_points=ref_pixels,
                slate_rectangle=None,
                skipped_points=None,
                updated_at=None,
                completed=True,
                label_studio_json=None,
                image_id=image_id,
                user_id=None,
            )
        )
        laser_labels_by_image_id[image_id] = LaserLabel(
            id=400 + i,
            label_studio_task_id=500 + i,
            label_studio_project_id=73,
            x=laser_pixel[0],
            y=laser_pixel[1],
            label="laser",
            updated_at=None,
            superseded=False,
            completed=True,
            label_studio_json=None,
            image_id=image_id,
            user_id=None,
        )

    return slate, intrinsics, dive_slate_labels, laser_labels_by_image_id


@pytest.mark.asyncio
async def test_returns_none_when_dive_has_no_slate(monkeypatch):
    dive = _dive(dive_slate_id=None)
    fs = _make_fs(dive, [], [], {}, _camera_intrinsics())
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.perform_laser_calibration_activity, 42
    )

    assert result is None
    fs.dives.put_laser_extrinsics.assert_not_called()


@pytest.mark.asyncio
async def test_returns_none_when_dive_has_no_slate_labels(monkeypatch):
    dive = _dive()
    fs = _make_fs(dive, [_slate()], [], {}, _camera_intrinsics())
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.perform_laser_calibration_activity, 42
    )

    assert result is None
    fs.dives.put_laser_extrinsics.assert_not_called()


@pytest.mark.asyncio
async def test_raises_when_too_few_usable_laser_points(monkeypatch):
    dive = _dive()
    slate, intrinsics, labels, lasers = _build_synthetic_scene(
        n_observations=1,
        laser_origin_world=np.array([-0.03, -0.10, 0.0]),
        laser_axis_world=np.array([0.0, 0.0, 1.0]),
        slate_distances=[0.5],
    )
    fs = _make_fs(dive, [slate], labels, lasers, intrinsics)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    with pytest.raises(ValueError, match="insufficient laser points"):
        await ActivityEnvironment().run(
            sut.perform_laser_calibration_activity, 42
        )

    fs.dives.put_laser_extrinsics.assert_not_called()


@pytest.mark.asyncio
async def test_recovers_known_laser_extrinsics_from_synthetic_scene(monkeypatch):
    laser_origin = np.array([-0.03, -0.10, 0.0])
    laser_axis = np.array([0.005, -0.02, 1.0])
    laser_axis = laser_axis / np.linalg.norm(laser_axis)

    dive = _dive()
    slate, intrinsics, labels, lasers = _build_synthetic_scene(
        n_observations=6,
        laser_origin_world=laser_origin,
        laser_axis_world=laser_axis,
        slate_distances=[0.40, 0.55, 0.70, 0.85, 1.00, 1.15],
    )
    fs = _make_fs(dive, [slate], labels, lasers, intrinsics)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.perform_laser_calibration_activity, 42
    )

    assert result == 999

    fs.dives.put_laser_extrinsics.assert_awaited_once()
    args, _ = fs.dives.put_laser_extrinsics.call_args
    written_dive_id, written_le = args
    assert written_dive_id == 42
    assert isinstance(written_le, LaserExtrinsics)
    assert written_le.dive_id == 42
    assert written_le.camera_id == 1

    fitted_axis = np.asarray(written_le.laser_axis, dtype=float)
    fitted_axis = fitted_axis / np.linalg.norm(fitted_axis)
    cos = float(np.clip(np.dot(fitted_axis, laser_axis), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(abs(cos))))
    assert angle_deg < 0.5

    fitted_pos = np.asarray(written_le.laser_position, dtype=float)
    pos_xy_l2 = float(np.linalg.norm(fitted_pos[:2] - laser_origin[:2]))
    assert pos_xy_l2 < 0.001
