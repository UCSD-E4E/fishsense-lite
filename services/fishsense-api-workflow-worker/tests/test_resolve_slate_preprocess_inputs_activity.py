# pylint: disable=unused-argument
"""Unit tests for resolve_slate_preprocess_inputs_activity."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.dive_slate import DiveSlate
from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.species_label import SpeciesLabel
from fishsense_api_workflow_worker.activities import (
    resolve_slate_preprocess_inputs_activity as sut,
)


_K = np.array([[3000.0, 0.0, 2048.0], [0.0, 3000.0, 1536.0], [0.0, 0.0, 1.0]])
_D = np.array([-0.05, 0.01, 0.0, 0.0, 0.0])
SLATE = "Slate, Laser on slate"


def _dive(*, camera_id: Optional[int] = 1, dive_slate_id: Optional[int] = 7) -> Dive:
    return Dive(
        id=42,
        name="d",
        path="/dev/null/42",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority="HIGH",
        flip_dive_slate=False,
        camera_id=camera_id,
        dive_slate_id=dive_slate_id,
    )


def _slate(slate_id: int = 7, *, dpi: int | None = 300, refs=None) -> DiveSlate:
    return DiveSlate(
        id=slate_id,
        name="test-slate",
        dpi=dpi,
        path="/dev/null",
        created_at=None,
        reference_points=refs if refs is not None else [(0.0, 0.0), (1.0, 1.0)],
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


def _species(image_id: int, *, content: str | None) -> SpeciesLabel:
    return SpeciesLabel(
        id=None,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=70,
        image_url=None,
        updated_at=None,
        completed=True,
        grouping=None,
        top_three_photos_of_group=None,
        slate_upside_down=None,
        laser_x=None,
        laser_y=None,
        laser_label=None,
        content_of_image=content,
        fish_measurable_category=None,
        fish_angle_category=None,
        fish_curved_category=None,
        label_studio_json={},
        image_id=image_id,
        user_id=None,
    )


def _slate_label(image_id: int, *, completed: bool) -> DiveSlateLabel:
    return DiveSlateLabel(
        id=None,
        image_id=image_id,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=66,
        image_url=None,
        updated_at=None,
        completed=completed,
        upside_down=None,
        reference_points=None,
        slate_rectangle=None,
        skipped_points=None,
        label_studio_json={},
        user_id=None,
    )


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_matrix=_K, distortion_coefficients=_D, camera_id=1
    )


def _make_fs(
    *,
    dive: Optional[Dive],
    intrinsics: Optional[CameraIntrinsics],
    slates: List[DiveSlate],
    images: List[Image],
    species: List[SpeciesLabel],
    slate_labels: List[DiveSlateLabel],
):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(return_value=dive)

    fs.cameras = MagicMock()
    fs.cameras.get_intrinsics = AsyncMock(return_value=intrinsics)

    fs.dive_slates = MagicMock()
    fs.dive_slates.get = AsyncMock(return_value=slates)

    fs.images = MagicMock()
    fs.images.get = AsyncMock(return_value=images)

    fs.labels = MagicMock()
    fs.labels.get_species_labels = AsyncMock(return_value=species)
    fs.labels.get_dive_slate_labels = AsyncMock(return_value=slate_labels)
    return fs


@pytest.mark.asyncio
async def test_returns_only_slate_marked_with_incomplete_slate_label(monkeypatch):
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        slates=[_slate(7)],
        images=[_image(1, "aaa"), _image(2, "bbb"), _image(3, "ccc")],
        species=[
            _species(1, content=SLATE),
            _species(2, content="Fish"),
            _species(3, content=SLATE),
        ],
        slate_labels=[_slate_label(1, completed=True)],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    result = await ActivityEnvironment().run(
        sut.resolve_slate_preprocess_inputs_activity, 42
    )
    # 1: slate-marked but completed -> dropped.
    # 2: not slate-marked -> dropped.
    # 3: slate-marked, no slate label -> kept.
    assert result.image_checksums == ["ccc"]
    assert result.dive_id == 42
    assert result.slate_id == 7
    assert result.slate_dpi == 300
    assert result.reference_points == [(0.0, 0.0), (1.0, 1.0)]
    assert result.camera_matrix == _K.tolist()
    assert result.distortion_coefficients == _D.tolist()


@pytest.mark.asyncio
async def test_raises_for_dive_without_dive_slate_id(monkeypatch):
    fs = _make_fs(
        dive=_dive(dive_slate_id=None),
        intrinsics=_intrinsics(),
        slates=[],
        images=[],
        species=[],
        slate_labels=[],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    with pytest.raises(ValueError, match="no dive_slate_id"):
        await ActivityEnvironment().run(
            sut.resolve_slate_preprocess_inputs_activity, 42
        )


@pytest.mark.asyncio
async def test_raises_when_slate_id_not_in_listing(monkeypatch):
    fs = _make_fs(
        dive=_dive(dive_slate_id=99),
        intrinsics=_intrinsics(),
        slates=[_slate(7)],
        images=[],
        species=[],
        slate_labels=[],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    with pytest.raises(ValueError, match="not found"):
        await ActivityEnvironment().run(
            sut.resolve_slate_preprocess_inputs_activity, 42
        )


@pytest.mark.asyncio
async def test_raises_when_slate_missing_dpi_or_refs(monkeypatch):
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        slates=[_slate(7, dpi=None)],
        images=[],
        species=[],
        slate_labels=[],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    with pytest.raises(ValueError, match="missing dpi or reference_points"):
        await ActivityEnvironment().run(
            sut.resolve_slate_preprocess_inputs_activity, 42
        )
