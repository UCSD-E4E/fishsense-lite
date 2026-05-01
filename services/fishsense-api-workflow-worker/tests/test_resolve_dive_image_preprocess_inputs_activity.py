# pylint: disable=unused-argument
"""Unit tests for resolve_dive_image_preprocess_inputs_activity.

Pins down:
  1. PREDICTION clusters are mapped to checksums in order.
  2. Cluster entries pointing at images that no longer exist are
     dropped (defensive — would otherwise 404 in download_raw).
  3. Camera intrinsics flatten from numpy to lists.
  4. Missing dive / camera_id / intrinsics raise ValueError.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_api_sdk.models.data_source import DataSource
from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.dive_frame_cluster import DiveFrameCluster
from fishsense_api_sdk.models.image import Image
from fishsense_api_workflow_worker.activities import (
    resolve_dive_image_preprocess_inputs_activity as sut,
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


def _cluster(cluster_id: int, image_ids: List[int]) -> DiveFrameCluster:
    return DiveFrameCluster(
        id=cluster_id,
        image_ids=image_ids,
        data_source=DataSource.PREDICTION,
        updated_at=None,
        dive_id=42,
        fish_id=None,
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
    clusters: List[DiveFrameCluster],
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
    fs.images.get_clusters = AsyncMock(return_value=clusters)
    return fs


@pytest.mark.asyncio
async def test_maps_clusters_to_checksums_in_order(monkeypatch):
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[_image(1, "aaa"), _image(2, "bbb"), _image(3, "ccc")],
        clusters=[_cluster(10, [1, 2]), _cluster(11, [3])],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_dive_image_preprocess_inputs_activity, 42
    )

    assert result.dive_id == 42
    assert result.clusters == [["aaa", "bbb"], ["ccc"]]
    assert result.camera_matrix == _K.tolist()
    assert result.distortion_coefficients == _D.tolist()


@pytest.mark.asyncio
async def test_drops_image_ids_that_no_longer_have_image_rows(monkeypatch):
    # Cluster references image 99, but the images endpoint doesn't
    # return one for it — drop it from the cluster checksums to avoid
    # a downstream 404.
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[_image(1, "aaa")],
        clusters=[_cluster(10, [1, 99])],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_dive_image_preprocess_inputs_activity, 42
    )
    assert result.clusters == [["aaa"]]


@pytest.mark.asyncio
async def test_drops_clusters_that_become_empty_after_image_id_filter(monkeypatch):
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[_image(1, "aaa")],
        clusters=[_cluster(10, [1]), _cluster(11, [99])],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_dive_image_preprocess_inputs_activity, 42
    )
    assert result.clusters == [["aaa"]]


@pytest.mark.asyncio
async def test_raises_when_dive_not_found(monkeypatch):
    fs = _make_fs(
        dive=None, intrinsics=_intrinsics(), images=[], clusters=[]
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    with pytest.raises(ValueError, match="not found"):
        await ActivityEnvironment().run(
            sut.resolve_dive_image_preprocess_inputs_activity, 42
        )


@pytest.mark.asyncio
async def test_raises_when_no_camera_id(monkeypatch):
    fs = _make_fs(
        dive=_dive(camera_id=None),
        intrinsics=_intrinsics(),
        images=[],
        clusters=[],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    with pytest.raises(ValueError, match="no camera_id"):
        await ActivityEnvironment().run(
            sut.resolve_dive_image_preprocess_inputs_activity, 42
        )


@pytest.mark.asyncio
async def test_raises_when_no_intrinsics(monkeypatch):
    fs = _make_fs(dive=_dive(), intrinsics=None, images=[], clusters=[])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    with pytest.raises(ValueError, match="no intrinsics"):
        await ActivityEnvironment().run(
            sut.resolve_dive_image_preprocess_inputs_activity, 42
        )
