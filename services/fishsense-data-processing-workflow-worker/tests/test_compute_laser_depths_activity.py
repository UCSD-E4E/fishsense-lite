"""Unit tests for compute_laser_depths_activity.

Records how far away the laser dot was, for every frame in a dive whose laser
label was validated — not just the measurable ones stage 14 visits. The
projection is the same one stage 14 uses (shared via `laser_geometry`, pinned
by `test_laser_geometry.py`); what this file covers is the plumbing around it:
which images get visited, which get skipped, and what the row records.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.laser_depth import LaserDepth
from fishsense_api_sdk.models.laser_extrinsics import LaserExtrinsics
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_data_processing_workflow_worker.activities import (
    compute_laser_depths_activity as sut,
)

CAMERA_MATRIX = np.array(
    [[3000.0, 0.0, 2048.0], [0.0, 3000.0, 1536.0], [0.0, 0.0, 1.0]]
)
CALIBRATION_ID = 11


def _camera_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_matrix=CAMERA_MATRIX, distortion_coefficients=np.zeros(5), camera_id=1
    )


def _dive(dive_id: int = 42, camera_id: int | None = 1) -> Dive:
    return Dive(
        id=dive_id,
        name=f"dive-{dive_id}",
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority="HIGH",
        flip_dive_slate=False,
        camera_id=camera_id,
        dive_slate_id=7,
    )


def _laser_extrinsics(extrinsics_id: int = CALIBRATION_ID) -> LaserExtrinsics:
    axis = np.array([0.0, -0.02, 1.0])
    return LaserExtrinsics(
        laser_position=np.array([-0.03, -0.10, 0.0]),
        laser_axis=axis / np.linalg.norm(axis),
        dive_id=42,
        camera_id=1,
        id=extrinsics_id,
    )


def _laser_pixel(laser_extrinsics: LaserExtrinsics, depth: float):
    """Where the laser dot lands when it hits a plane at `depth`."""
    origin = np.asarray(laser_extrinsics.laser_position, dtype=float)
    axis = np.asarray(laser_extrinsics.laser_axis, dtype=float)
    hit = origin + ((depth - origin[2]) / axis[2]) * axis
    projected = CAMERA_MATRIX @ hit
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def _laser_label(
    label_id: int, image_id: int, x, y, *, completed=True, superseded=False
) -> LaserLabel:
    return LaserLabel(
        id=label_id,
        label_studio_task_id=None,
        label_studio_project_id=73,
        x=x,
        y=y,
        label=None,
        updated_at=None,
        superseded=superseded,
        completed=completed,
        label_studio_json=None,
        image_id=image_id,
        user_id=None,
    )


# `None` is a meaningful value for the first three (it is what the SDK returns
# for "this dive has no calibration"), so the default has to be distinguishable
# from an explicit None.
_DEFAULT = object()


def _make_fs(
    *,
    dive=_DEFAULT,
    intrinsics=_DEFAULT,
    laser_extrinsics=_DEFAULT,
    laser_labels: list[LaserLabel] | None = None,
    existing_depths: list[LaserDepth] | None = None,
):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(return_value=_dive() if dive is _DEFAULT else dive)
    fs.dives.get_laser_extrinsics = AsyncMock(
        return_value=(
            _laser_extrinsics() if laser_extrinsics is _DEFAULT else laser_extrinsics
        )
    )

    fs.cameras = MagicMock()
    fs.cameras.get_intrinsics = AsyncMock(
        return_value=_camera_intrinsics() if intrinsics is _DEFAULT else intrinsics
    )

    fs.labels = MagicMock()
    fs.labels.get_laser_labels = AsyncMock(return_value=laser_labels)

    fs.images = MagicMock()
    fs.images.get_laser_depths = AsyncMock(return_value=existing_depths or [])
    fs.images.put_laser_depth = AsyncMock(return_value=1)
    return fs


async def _run(monkeypatch, fs, dive_id: int = 42):
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    return await ActivityEnvironment().run(sut.compute_laser_depths_activity, dive_id)


@pytest.mark.asyncio
async def test_records_depth_range_and_provenance(monkeypatch):
    extrinsics = _laser_extrinsics()
    x, y = _laser_pixel(extrinsics, 1.20)
    fs = _make_fs(
        laser_extrinsics=extrinsics,
        laser_labels=[_laser_label(101, 100, x, y)],
    )

    result = await _run(monkeypatch, fs)

    assert result.computed == 1
    image_id, depth = fs.images.put_laser_depth.call_args.args
    assert image_id == 100
    assert depth.depth_m == pytest.approx(1.20, abs=1e-3)
    # Off-axis dot, so the slant distance is strictly longer than the depth.
    assert depth.range_m > depth.depth_m
    assert (depth.laser_label_id, depth.laser_extrinsics_id) == (101, extrinsics.id)
    # The dot was constructed on the laser ray, so the two rays meet and the
    # closest-approach distance collapses to the float32 noise floor.
    assert depth.residual_m == pytest.approx(0.0, abs=1e-5)


@pytest.mark.asyncio
async def test_records_the_residual_for_an_inconsistent_dot(monkeypatch):
    """A dot displaced across the laser's epipolar line cannot lie on the
    laser at any depth. The depth is still finite and positive — nothing else
    in the pipeline would question it — and the residual is the only thing
    that says the label and the calibration disagree.

    Stored, not gated: a threshold has to come from the observed distribution,
    and the residual is metric, so the same number means different things at
    0.9 m and 2.5 m."""
    extrinsics = _laser_extrinsics()
    x, y = _laser_pixel(extrinsics, 1.20)
    fs = _make_fs(
        laser_extrinsics=extrinsics,
        laser_labels=[_laser_label(101, 100, x, y + 150.0)],
    )

    result = await _run(monkeypatch, fs)

    assert result.computed == 1
    _, depth = fs.images.put_laser_depth.call_args.args
    assert depth.depth_m > 0.0
    assert depth.residual_m > 1e-3


@pytest.mark.asyncio
async def test_skips_images_already_current(monkeypatch):
    """Drains: the same label and the same calibration produce the same
    number, so recomputing it is pure cost."""
    extrinsics = _laser_extrinsics()
    x, y = _laser_pixel(extrinsics, 1.20)
    fs = _make_fs(
        laser_extrinsics=extrinsics,
        laser_labels=[_laser_label(101, 100, x, y)],
        existing_depths=[
            LaserDepth(
                id=1,
                depth_m=1.20,
                range_m=1.21,
                image_id=100,
                laser_label_id=101,
                laser_extrinsics_id=extrinsics.id,
            )
        ],
    )

    result = await _run(monkeypatch, fs)

    fs.images.put_laser_depth.assert_not_awaited()
    assert (result.computed, result.skipped_current) == (0, 1)


@pytest.mark.asyncio
async def test_recomputes_after_a_recalibration(monkeypatch):
    extrinsics = _laser_extrinsics(extrinsics_id=12)
    x, y = _laser_pixel(extrinsics, 1.20)
    fs = _make_fs(
        laser_extrinsics=extrinsics,
        laser_labels=[_laser_label(101, 100, x, y)],
        existing_depths=[
            LaserDepth(
                id=1,
                depth_m=1.20,
                image_id=100,
                laser_label_id=101,
                laser_extrinsics_id=11,
            )
        ],
    )

    result = await _run(monkeypatch, fs)

    assert result.computed == 1
    assert fs.images.put_laser_depth.call_args.args[1].laser_extrinsics_id == 12


@pytest.mark.asyncio
async def test_recomputes_after_the_label_is_replaced(monkeypatch):
    extrinsics = _laser_extrinsics()
    x, y = _laser_pixel(extrinsics, 1.20)
    fs = _make_fs(
        laser_extrinsics=extrinsics,
        laser_labels=[_laser_label(102, 100, x, y)],
        existing_depths=[
            LaserDepth(
                id=1,
                depth_m=9.99,
                image_id=100,
                laser_label_id=101,
                laser_extrinsics_id=extrinsics.id,
            )
        ],
    )

    result = await _run(monkeypatch, fs)

    assert result.computed == 1
    assert fs.images.put_laser_depth.call_args.args[1].laser_label_id == 102


@pytest.mark.parametrize(
    "label_kwargs",
    [
        {"completed": False},
        {"superseded": True},
    ],
    ids=["incomplete", "superseded"],
)
@pytest.mark.asyncio
async def test_ignores_labels_that_are_not_a_validated_fix(monkeypatch, label_kwargs):
    """The same `valid laser` gate stages 1/2/5.1/14 cascade from. A dot a
    labeler placed but nobody validated, or one RANSAC superseded, is not a
    position to derive a distance from."""
    extrinsics = _laser_extrinsics()
    x, y = _laser_pixel(extrinsics, 1.20)
    fs = _make_fs(
        laser_extrinsics=extrinsics,
        laser_labels=[_laser_label(101, 100, x, y, **label_kwargs)],
    )

    result = await _run(monkeypatch, fs)

    fs.images.put_laser_depth.assert_not_awaited()
    assert result.skipped_unusable_label == 1


@pytest.mark.asyncio
async def test_ignores_labels_missing_a_coordinate(monkeypatch):
    fs = _make_fs(laser_labels=[_laser_label(101, 100, None, None)])

    result = await _run(monkeypatch, fs)

    fs.images.put_laser_depth.assert_not_awaited()
    assert result.skipped_unusable_label == 1


@pytest.mark.asyncio
async def test_refuses_to_store_an_impossible_depth(monkeypatch):
    """The kernel signals "these rays do not meet" with a zero or negative Z
    rather than NaN, and a negative depth yields a perfectly ordinary-looking
    length downstream (see test_laser_geometry). Storing one would launder a
    bad label into a plausible distance."""
    extrinsics = _laser_extrinsics()
    # Laser offset is -x, so a real dot lands left of the principal point;
    # a label to the right of it inverts the solve.
    fs = _make_fs(
        laser_extrinsics=extrinsics,
        laser_labels=[_laser_label(101, 100, 3000.0, 1600.0)],
    )

    result = await _run(monkeypatch, fs)

    fs.images.put_laser_depth.assert_not_awaited()
    assert result.skipped_invalid_geometry == 1


@pytest.mark.asyncio
async def test_handles_a_dive_with_no_laser_labels(monkeypatch):
    """`None` is the SDK's "no labels" signal (a 404), not an error."""
    fs = _make_fs(laser_labels=None)

    result = await _run(monkeypatch, fs)

    assert result.computed == 0
    fs.images.put_laser_depth.assert_not_awaited()


@pytest.mark.asyncio
async def test_raises_when_the_dive_has_no_calibration(monkeypatch):
    """Fail loud rather than skip: the cohort selector only offers dives with
    resolvable extrinsics, so arriving here without them means the selector
    and the activity disagree — which is how a dive silently never drains."""
    fs = _make_fs(laser_extrinsics=None, laser_labels=[])

    with pytest.raises(ValueError, match="laser_extrinsics"):
        await _run(monkeypatch, fs)


@pytest.mark.asyncio
async def test_raises_when_the_camera_has_no_intrinsics(monkeypatch):
    fs = _make_fs(intrinsics=None, laser_labels=[])

    with pytest.raises(ValueError, match="intrinsics"):
        await _run(monkeypatch, fs)


@pytest.mark.asyncio
async def test_fetches_existing_depths_once_per_dive(monkeypatch):
    """One dive-scoped read, not a per-image lookup — the same shape stage 14
    uses for measurements."""
    extrinsics = _laser_extrinsics()
    labels = [
        _laser_label(100 + i, i, *_laser_pixel(extrinsics, 1.0 + i / 10))
        for i in range(4)
    ]
    fs = _make_fs(laser_extrinsics=extrinsics, laser_labels=labels)

    result = await _run(monkeypatch, fs)

    assert result.computed == 4
    fs.images.get_laser_depths.assert_awaited_once_with(42)


@pytest.mark.asyncio
async def test_duplicate_valid_labels_produce_one_depth_per_image(monkeypatch):
    """An image can carry several valid laser labels — 461 prod images do,
    nearly all duplicates of the same dot. `LaserDepth` is one row per image,
    so processing every label triangulates the same dot twice and leaves
    whichever ran last recorded. Take the lowest label id and move on: half
    the work, and the recorded provenance is stable across re-runs instead of
    flapping with iteration order."""
    extrinsics = _laser_extrinsics()
    x, y = _laser_pixel(extrinsics, 1.20)
    fs = _make_fs(
        laser_extrinsics=extrinsics,
        laser_labels=[
            _laser_label(102, 100, x, y),
            _laser_label(101, 100, x, y),
        ],
    )

    result = await _run(monkeypatch, fs)

    assert result.computed == 1
    assert fs.images.put_laser_depth.await_count == 1
    assert fs.images.put_laser_depth.call_args.args[1].laser_label_id == 101
