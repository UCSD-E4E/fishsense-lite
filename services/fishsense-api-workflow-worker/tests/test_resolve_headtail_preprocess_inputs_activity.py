# pylint: disable=unused-argument
"""Unit tests for resolve_headtail_preprocess_inputs_activity.

The headtail cohort cascades from valid laser labels (completed, not
superseded, both x/y populated). This resolver mirrors the API SQL
predicate so the per-image work matches what the cohort selector
promised.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.headtail_label import HeadTailLabel
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_workflow_worker.activities import (
    resolve_headtail_preprocess_inputs_activity as sut,
)


_K = np.array([[3000.0, 0.0, 2048.0], [0.0, 3000.0, 1536.0], [0.0, 0.0, 1.0]])
_D = np.array([-0.05, 0.01, 0.0, 0.0, 0.0])


def _dive(*, camera_id: Optional[int] = 1) -> Dive:
    return Dive(
        id=42,
        name="d",
        path="/dev/null/42",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority="HIGH",
        flip_dive_slate=False,
        camera_id=camera_id,
        dive_slate_id=None,
    )


def _image(image_id: int, checksum: str) -> Image:
    return Image(
        id=image_id,
        path=f"/dev/null/{image_id}",
        taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        checksum=checksum,
        is_canonical=True,
        dive_id=42,
        camera_id=1,
    )


def _laser(
    image_id: int,
    *,
    completed: bool = True,
    superseded: bool = False,
    x: Optional[float] = 100.0,
    y: Optional[float] = 200.0,
) -> LaserLabel:
    return LaserLabel(
        id=image_id * 7,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=43,
        x=x,
        y=y,
        label="laser",
        updated_at=None,
        superseded=superseded,
        completed=completed,
        label_studio_json={},
        image_id=image_id,
        user_id=None,
    )


def _headtail(
    image_id: int,
    *,
    completed: bool,
    project_id: Optional[int] = 71,
) -> HeadTailLabel:
    return HeadTailLabel(
        id=None,
        image_id=image_id,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=project_id,
        image_url=None,
        updated_at=None,
        completed=completed,
        head_x=None,
        head_y=None,
        tail_x=None,
        tail_y=None,
        label_studio_json={},
        user_id=None,
        superseded=False,
    )


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_matrix=_K, distortion_coefficients=_D, camera_id=1
    )


def _make_fs(
    *,
    dive: Optional[Dive],
    intrinsics: Optional[CameraIntrinsics],
    images: List[Image],
    laser: List[LaserLabel],
    headtail: List[HeadTailLabel],
):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(return_value=dive)

    fs.cameras = MagicMock()
    fs.cameras.get_intrinsics = AsyncMock(return_value=intrinsics)

    fs.images = MagicMock()
    fs.images.get = AsyncMock(return_value=images)

    fs.labels = MagicMock()
    fs.labels.get_laser_labels = AsyncMock(return_value=laser)
    fs.labels.get_headtail_labels = AsyncMock(return_value=headtail)
    return fs


@pytest.mark.asyncio
async def test_returns_only_valid_laser_without_any_real_headtail(monkeypatch):
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[
            _image(1, "aaa"),
            _image(2, "bbb"),
            _image(3, "ccc"),
            _image(4, "ddd"),
        ],
        laser=[
            _laser(1),  # valid + has completed headtail -> dropped
            _laser(2, completed=False),  # incomplete laser -> dropped
            _laser(3),  # valid + no headtail -> kept
            _laser(4),  # valid + incomplete real headtail -> dropped
        ],
        headtail=[
            _headtail(1, completed=True),
            _headtail(4, completed=False),
        ],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_headtail_preprocess_inputs_activity, 42
    )

    assert result.image_checksums == ["ccc"]
    assert result.dive_id == 42
    assert result.camera_matrix == _K.tolist()
    assert result.distortion_coefficients == _D.tolist()


@pytest.mark.asyncio
async def test_image_with_only_null_project_sentinel_treated_as_unlabeled(
    monkeypatch,
):
    """NULL-`project_id` HeadTailLabel rows are legacy sentinels —
    they must NOT exclude a laser-cascaded image. Matches the API
    selector predicate."""
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[_image(1, "aaa"), _image(2, "bbb")],
        laser=[_laser(1), _laser(2)],
        headtail=[
            _headtail(1, completed=False, project_id=None),
            _headtail(2, completed=False, project_id=71),
        ],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_headtail_preprocess_inputs_activity, 42
    )

    # Image 1: only a sentinel -> still needs work.
    # Image 2: real-project row -> excluded.
    assert result.image_checksums == ["aaa"]


@pytest.mark.asyncio
async def test_drops_lasers_that_are_incomplete_superseded_or_null_xy(
    monkeypatch,
):
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[
            _image(1, "aaa"),
            _image(2, "bbb"),
            _image(3, "ccc"),
            _image(4, "ddd"),
        ],
        laser=[
            _laser(1, completed=False),
            _laser(2, superseded=True),
            _laser(3, x=None),
            _laser(4, y=None),
        ],
        headtail=[],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_headtail_preprocess_inputs_activity, 42
    )
    assert result.image_checksums == []


@pytest.mark.asyncio
async def test_empty_when_no_laser_labels(monkeypatch):
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[_image(1, "aaa")],
        laser=[],
        headtail=[],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    result = await ActivityEnvironment().run(
        sut.resolve_headtail_preprocess_inputs_activity, 42
    )
    assert result.image_checksums == []


@pytest.mark.asyncio
async def test_raises_when_dive_missing_or_camera_missing_or_intrinsics_missing(
    monkeypatch,
):
    cases = [
        (_make_fs(
            dive=None, intrinsics=_intrinsics(),
            images=[], laser=[], headtail=[],
        ), "not found"),
        (_make_fs(
            dive=_dive(camera_id=None), intrinsics=_intrinsics(),
            images=[], laser=[], headtail=[],
        ), "no camera_id"),
        (_make_fs(
            dive=_dive(), intrinsics=None,
            images=[], laser=[], headtail=[],
        ), "no intrinsics"),
    ]
    for fs, match in cases:
        monkeypatch.setattr(sut, "get_fs_client", lambda fs=fs: fs)
        with pytest.raises(ValueError, match=match):
            await ActivityEnvironment().run(
                sut.resolve_headtail_preprocess_inputs_activity, 42
            )
