# pylint: disable=unused-argument
"""Unit tests for resolve_dive_frame_clustering_inputs_activity."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.image import Image
from fishsense_api_workflow_worker.activities import (
    resolve_dive_frame_clustering_inputs_activity as sut,
)


_BASE = datetime(2026, 5, 5, 10, 0, 0, tzinfo=timezone.utc)


def _image(image_id: int, *, offset_seconds: int = 0) -> Image:
    return Image(
        id=image_id,
        path=f"/dev/null/{image_id}",
        taken_datetime=_BASE + timedelta(seconds=offset_seconds),
        checksum=f"chk{image_id}",
        is_canonical=True,
        dive_id=42,
        camera_id=1,
    )


def _make_fs(images: List[Image]):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.images = MagicMock()
    fs.images.get = AsyncMock(return_value=images)
    return fs


@pytest.mark.asyncio
async def test_returns_image_id_taken_datetime_pairs(monkeypatch):
    images = [_image(1, offset_seconds=0), _image(2, offset_seconds=5)]
    monkeypatch.setattr(sut, "get_fs_client", lambda: _make_fs(images))

    result = await ActivityEnvironment().run(
        sut.resolve_dive_frame_clustering_inputs_activity, 42
    )

    assert result.dive_id == 42
    assert [img.image_id for img in result.images] == [1, 2]
    assert result.images[0].taken_datetime == _BASE
    assert result.images[1].taken_datetime == _BASE + timedelta(seconds=5)


@pytest.mark.asyncio
async def test_returns_empty_image_list_when_dive_has_no_images(monkeypatch):
    monkeypatch.setattr(sut, "get_fs_client", lambda: _make_fs([]))

    result = await ActivityEnvironment().run(
        sut.resolve_dive_frame_clustering_inputs_activity, 42
    )

    assert result.dive_id == 42
    assert result.images == []


@pytest.mark.asyncio
async def test_drops_images_with_null_id(monkeypatch):
    """SDK Image.id is `int | None` — drop the None ones defensively
    so the data-worker DTO carries only addressable ids."""
    images = [_image(1), Image(  # noqa: E501
        id=None,
        path="/dev/null/x",
        taken_datetime=_BASE,
        checksum="x",
        is_canonical=True,
        dive_id=42,
        camera_id=1,
    )]
    monkeypatch.setattr(sut, "get_fs_client", lambda: _make_fs(images))

    result = await ActivityEnvironment().run(
        sut.resolve_dive_frame_clustering_inputs_activity, 42
    )

    assert [img.image_id for img in result.images] == [1]
