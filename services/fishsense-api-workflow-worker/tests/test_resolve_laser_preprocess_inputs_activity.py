# pylint: disable=unused-argument,protected-access
"""Unit tests for resolve_laser_preprocess_inputs_activity.

Pins down:
  1. Returns only checksums of images that have no LaserLabel row at
     all (in any project). Once populate seeds even an incomplete
     row, the image's preprocessed JPEG is on the file-exchange and
     the resolver must not return it. Matches the API selector
     predicate.
  2. Camera intrinsics are flattened from numpy to lists.
  3. Default bbox + laser region land in the resolved input.
  3b. An image whose label is flagged `needs_reprocess` comes back even
     though it is labelled -- the resolver must mirror the selector, or
     the dive is picked and then found to have no work, every hour.
  4. Missing dive / camera_id / intrinsics raise ValueError so the
     parent workflow surfaces the data problem instead of silently
     dispatching 0-image work to the data-worker.
"""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_workflow_worker.activities import (
    resolve_laser_preprocess_inputs_activity as sut,
)

# Shared with the other resolver suites — see `worker_tests_support.preprocess`.
from worker_tests_support.preprocess import (  # noqa: F401
    CAMERA_MATRIX as _K,
    DISTORTION as _D,
    dive as _dive,
    image as _image,
    laser_label as _label,
    intrinsics as _intrinsics,
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

    # Image 1 has a completed label -> excluded.
    # Image 2 has an incomplete label -> excluded (any row excludes).
    # Image 3 has no label at all -> included.
    assert result.dive_id == 42
    assert set(result.image_checksums) == {"ccc"}
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

    assert not result.image_checksums


@pytest.mark.asyncio
async def test_returns_empty_checksums_when_only_incomplete_labels(monkeypatch):
    """Steady-state after populate seeds incomplete sentinel rows: every
    image has at least one row, so the resolver must return no work
    even though no label is completed yet. Mirrors the API selector's
    drop-out predicate."""
    images = [_image(1, "aaa"), _image(2, "bbb")]
    labels = [_label(1, completed=False), _label(2, completed=False)]
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

    assert not result.image_checksums


@pytest.mark.asyncio
async def test_image_with_completed_and_incomplete_rows_treated_as_labeled(
    monkeypatch,
):
    """Multi-row state: one image carries a completed row in project
    43, an incomplete sentinel in project NULL, plus an incomplete
    real-project row. Sentinels are ignored but the real-project
    rows are sufficient to exclude.
    """
    images = [_image(1, "aaa"), _image(2, "bbb")]
    labels = [
        # Image 1: completed real-project row + sentinel (sentinel ignored).
        _label(1, completed=True, project_id=43),
        _label(1, completed=False, project_id=None),
        # Image 2: incomplete real-project row.
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

    # Both images carry at least one non-sentinel row -> both excluded.
    assert not result.image_checksums


@pytest.mark.asyncio
async def test_image_with_only_null_project_sentinel_treated_as_unlabeled(
    monkeypatch,
):
    """NULL-`project_id` rows are legacy sentinels (~2000 of them in
    prod). The resolver must ignore them when deciding whether an
    image needs preprocessing — otherwise prod's existing sentinel
    population would permanently drain the work set. Mirrors the
    API selector predicate.
    """
    images = [_image(1, "aaa"), _image(2, "bbb")]
    labels = [
        # Image 1: only a sentinel row -> needs work.
        _label(1, completed=False, project_id=None),
        # Image 2: real-project incomplete row -> excluded.
        _label(2, completed=False, project_id=99),
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

    assert result.image_checksums == ["aaa"]


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


def test_flagged_image_is_returned_though_labelled():
    """`needs_reprocess` overrides "already has a row".

    The selector picks a dive on this flag; if the resolver did not agree,
    the parent would stage that dive's raw `.ORF`s from the NAS, resolve
    zero images, and repeat on every hourly firing while the flag stayed up.
    """
    images = [_image(1, "a"), _image(2, "b")]
    labels = [
        _label(1, completed=True, needs_reprocess=True),
        _label(2, completed=True, needs_reprocess=False),
    ]
    assert [i.id for i in sut._select_unlabeled_images(images, labels)] == [1]


def test_unflagged_fully_labelled_set_is_empty():
    """The unchanged default -- without a flag a labelled image stays out."""
    images = [_image(1, "a")]
    labels = [_label(1, completed=True)]
    assert not sut._select_unlabeled_images(images, labels)


def test_unlabelled_and_flagged_images_are_both_returned():
    images = [_image(1, "a"), _image(2, "b"), _image(3, "c")]
    labels = [
        _label(1, completed=True, needs_reprocess=True),
        _label(2, completed=True, needs_reprocess=False),
    ]
    assert [i.id for i in sut._select_unlabeled_images(images, labels)] == [1, 3]


def test_bbox_is_the_region_polygons_bounding_box():
    """The data-worker draws `laser_region` and falls back to `bbox`; the
    resolver has to send both, and they have to agree."""
    assert sut.LASER_REGION_POLYGON
    assert sut.DEFAULT_LASER_BBOX == [
        min(v[0] for v in sut.LASER_REGION_POLYGON),
        min(v[1] for v in sut.LASER_REGION_POLYGON),
        max(v[0] for v in sut.LASER_REGION_POLYGON),
        max(v[1] for v in sut.LASER_REGION_POLYGON),
    ]
