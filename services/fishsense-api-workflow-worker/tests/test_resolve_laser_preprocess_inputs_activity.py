# pylint: disable=unused-argument
"""Unit tests for resolve_laser_preprocess_inputs_activity.

Pins down:
  1. Returns only checksums of images that have no completed
     LaserLabel in any project — multi-row state from prod (one
     completed row in project 43, one incomplete sentinel in project
     NULL) correctly resolves to "image is labeled". The earlier
     dict-collapse filter could resolve either way depending on SDK
     iteration order.
  2. Camera intrinsics are flattened from numpy to lists.
  3. Default bbox lands in the resolved input.
  4. Missing dive / camera_id / intrinsics raise ValueError so the
     parent workflow surfaces the data problem instead of silently
     dispatching 0-image work to the data-worker.
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
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_workflow_worker.activities import (
    resolve_laser_preprocess_inputs_activity as sut,
)


_K = np.array([[3000.0, 0.0, 2048.0], [0.0, 3000.0, 1536.0], [0.0, 0.0, 1.0]])
_D = np.array([-0.05, 0.01, 0.0, 0.0, 0.0])


def _dive(dive_id: int = 42, *, camera_id: Optional[int] = 1) -> Dive:
    return Dive(
        id=dive_id,
        name=f"dive-{dive_id}",
        path=f"/dev/null/{dive_id}",
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


def _label(
    image_id: int, *, completed: bool, project_id: Optional[int] = 73
) -> LaserLabel:
    return LaserLabel(
        id=None,
        image_id=image_id,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=project_id,
        updated_at=None,
        completed=completed,
        label_studio_json={},
        user_id=None,
        superseded=False,
        x=None,
        y=None,
        label=None,
    )


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_matrix=_K,
        distortion_coefficients=_D,
        camera_id=1,
    )


def _make_fs(
    *,
    dive: Optional[Dive],
    intrinsics: Optional[CameraIntrinsics],
    images: List[Image],
    laser_labels: List[LaserLabel],
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
    fs.labels.get_laser_labels = AsyncMock(return_value=laser_labels)
    return fs


@pytest.mark.asyncio
async def test_returns_only_unlabeled_image_checksums(monkeypatch):
    images = [_image(1, "aaa"), _image(2, "bbb"), _image(3, "ccc")]
    labels = [_label(1, completed=True), _label(2, completed=False)]
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=images,
        laser_labels=labels,
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_laser_preprocess_inputs_activity, 42
    )

    # Image 1 has a completed label -> excluded. Image 2 has only an
    # incomplete label, image 3 has no label at all -> both included.
    assert result.dive_id == 42
    assert set(result.image_checksums) == {"bbb", "ccc"}
    assert result.camera_matrix == _K.tolist()
    assert result.distortion_coefficients == _D.tolist()
    assert result.bbox == sut.DEFAULT_LASER_BBOX
    # Defensive copy: the resolver must not hand back the module-level
    # constant by reference, since pydantic doesn't deep-copy on input.
    assert result.bbox is not sut.DEFAULT_LASER_BBOX


@pytest.mark.asyncio
async def test_returns_empty_checksums_when_all_labels_completed(monkeypatch):
    images = [_image(1, "aaa"), _image(2, "bbb")]
    labels = [_label(1, completed=True), _label(2, completed=True)]
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=images,
        laser_labels=labels,
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_laser_preprocess_inputs_activity, 42
    )

    assert result.image_checksums == []


@pytest.mark.asyncio
async def test_image_with_completed_and_incomplete_rows_treated_as_labeled(
    monkeypatch,
):
    """Mirrors the prod state on dive 393: each image carries a
    completed row in project 43 plus an incomplete sentinel row in
    project NULL. Any one completed row is enough to count the image
    as labeled, regardless of SDK iteration order."""
    images = [_image(1, "aaa"), _image(2, "bbb")]
    labels = [
        # Image 1: completed in 43, also has an incomplete sentinel.
        # Order: completed first, sentinel second — would have lost in
        # the previous dict-collapse filter and leaked image 1 into
        # the work set.
        _label(1, completed=True, project_id=43),
        _label(1, completed=False, project_id=None),
        # Image 2: only an incomplete row -> still needs work.
        _label(2, completed=False, project_id=43),
    ]
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=images,
        laser_labels=labels,
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_laser_preprocess_inputs_activity, 42
    )

    assert result.image_checksums == ["bbb"]


@pytest.mark.asyncio
async def test_raises_when_dive_not_found(monkeypatch):
    fs = _make_fs(
        dive=None, intrinsics=_intrinsics(), images=[], laser_labels=[]
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    with pytest.raises(ValueError, match="not found"):
        await ActivityEnvironment().run(
            sut.resolve_laser_preprocess_inputs_activity, 42
        )


@pytest.mark.asyncio
async def test_raises_when_dive_has_no_camera_id(monkeypatch):
    fs = _make_fs(
        dive=_dive(camera_id=None),
        intrinsics=_intrinsics(),
        images=[],
        laser_labels=[],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    with pytest.raises(ValueError, match="no camera_id"):
        await ActivityEnvironment().run(
            sut.resolve_laser_preprocess_inputs_activity, 42
        )


@pytest.mark.asyncio
async def test_raises_when_camera_has_no_intrinsics(monkeypatch):
    fs = _make_fs(
        dive=_dive(), intrinsics=None, images=[], laser_labels=[]
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    with pytest.raises(ValueError, match="no intrinsics"):
        await ActivityEnvironment().run(
            sut.resolve_laser_preprocess_inputs_activity, 42
        )
