# pylint: disable=unused-argument
"""Unit tests for resolve_headtail_preprocess_inputs_activity."""

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
from fishsense_api_sdk.models.species_label import SpeciesLabel
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


def _species(image_id: int, *, top_three: bool) -> SpeciesLabel:
    return SpeciesLabel(
        id=None,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=70,
        image_url=None,
        updated_at=None,
        completed=True,
        grouping=None,
        top_three_photos_of_group=top_three,
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
    species: List[SpeciesLabel],
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
    fs.labels.get_species_labels = AsyncMock(return_value=species)
    fs.labels.get_headtail_labels = AsyncMock(return_value=headtail)
    return fs


@pytest.mark.asyncio
async def test_returns_only_top_three_without_any_headtail(monkeypatch):
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[
            _image(1, "aaa"),
            _image(2, "bbb"),
            _image(3, "ccc"),
            _image(4, "ddd"),
        ],
        species=[
            _species(1, top_three=True),
            _species(2, top_three=False),
            _species(3, top_three=True),
            _species(4, top_three=True),
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

    # 1: top-three but headtail row exists (completed) -> dropped.
    # 2: not top-three -> dropped.
    # 3: top-three + no headtail row -> kept.
    # 4: top-three + headtail row exists (incomplete) -> dropped (any
    #    row excludes — matches API selector predicate).
    assert result.image_checksums == ["ccc"]
    assert result.dive_id == 42
    assert result.camera_matrix == _K.tolist()
    assert result.distortion_coefficients == _D.tolist()


@pytest.mark.asyncio
async def test_image_with_only_null_project_sentinel_treated_as_unlabeled(
    monkeypatch,
):
    """NULL-`project_id` HeadTailLabel rows are legacy sentinels — the
    resolver must ignore them when deciding whether a top-three image
    needs preprocessing. Matches the API selector predicate."""
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[_image(1, "aaa"), _image(2, "bbb")],
        species=[
            _species(1, top_three=True),
            _species(2, top_three=True),
        ],
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
async def test_empty_when_no_top_three(monkeypatch):
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[_image(1, "aaa")],
        species=[_species(1, top_three=False)],
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
            images=[], species=[], headtail=[],
        ), "not found"),
        (_make_fs(
            dive=_dive(camera_id=None), intrinsics=_intrinsics(),
            images=[], species=[], headtail=[],
        ), "no camera_id"),
        (_make_fs(
            dive=_dive(), intrinsics=None,
            images=[], species=[], headtail=[],
        ), "no intrinsics"),
    ]
    for fs, match in cases:
        monkeypatch.setattr(sut, "get_fs_client", lambda fs=fs: fs)
        with pytest.raises(ValueError, match=match):
            await ActivityEnvironment().run(
                sut.resolve_headtail_preprocess_inputs_activity, 42
            )
