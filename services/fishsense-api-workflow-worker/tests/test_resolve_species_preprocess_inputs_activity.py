# pylint: disable=unused-argument
"""Unit tests for resolve_species_preprocess_inputs_activity.

The species cohort cascades from valid laser labels (completed, not
superseded, both x/y populated) AND requires a PREDICTION cluster
gate (so stage 1 has run). Resolver mirrors the API SQL predicate at
image granularity: cluster.image_ids are filtered to images whose
laser is valid AND whose species label is missing or only a NULL-
project sentinel.

Pins down:
  1. PREDICTION clusters are mapped to checksums in order, filtered
     to laser-valid + species-unlabeled images.
  2. Images carrying a non-sentinel SpeciesLabel are dropped (so a
     repopulate doesn't double-import LS tasks).
  3. NULL-project SpeciesLabel sentinels do NOT drop an image.
  4. Cluster entries pointing at images that no longer exist are
     dropped (defensive — would otherwise 404 in download_raw).
  5. Camera intrinsics flatten from numpy to lists.
  6. Missing dive / camera_id / intrinsics raise ValueError.
"""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_api_sdk.models.data_source import DataSource
from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.dive_frame_cluster import DiveFrameCluster
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_sdk.models.species_label import SpeciesLabel
from fishsense_api_workflow_worker.activities import (
    resolve_species_preprocess_inputs_activity as sut,
)

# Shared with the other resolver suites — see `worker_tests_support.preprocess`.
from worker_tests_support.preprocess import (  # noqa: F401
    CAMERA_MATRIX as _K,
    DISTORTION as _D,
    dive as _dive,
    image as _image,
    intrinsics as _intrinsics,
)
# The same laser-label builder the populate suites use.
from worker_tests_support.populate import laser_label as _laser




def _cluster(cluster_id: int, image_ids: List[int]) -> DiveFrameCluster:
    return DiveFrameCluster(
        id=cluster_id,
        image_ids=image_ids,
        data_source=DataSource.PREDICTION,
        updated_at=None,
        dive_id=42,
        fish_id=None,
    )


def _species(
    image_id: int,
    *,
    completed: bool = False,
    project_id: Optional[int] = 70,
) -> SpeciesLabel:
    return SpeciesLabel(
        id=image_id * 5,
        image_id=image_id,
        label_studio_task_id=image_id * 11,
        label_studio_project_id=project_id,
        image_url=f"http://example.com/{image_id}.JPG",
        updated_at=None,
        completed=completed,
        superseded=False,
        label_studio_json={},
        user_id=None,
        grouping=None,
        top_three_photos_of_group=None,
        slate_upside_down=None,
        laser_x=None,
        laser_y=None,
        laser_label=None,
        content_of_image=None,
        fish_measurable_category=None,
        fish_angle_category=None,
        fish_curved_category=None,
    )


def _make_fs(
    *,
    dive: Optional[Dive],
    intrinsics: Optional[CameraIntrinsics],
    images: List[Image],
    clusters: List[DiveFrameCluster],
    laser_labels: Optional[List[LaserLabel]] = None,
    species_labels: Optional[List[SpeciesLabel]] = None,
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

    fs.labels = MagicMock()
    fs.labels.get_laser_labels = AsyncMock(return_value=laser_labels or [])
    fs.labels.get_species_labels = AsyncMock(return_value=species_labels or [])
    return fs


@pytest.mark.asyncio
async def test_keeps_laser_valid_unlabeled_images_in_cluster_order(monkeypatch):
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[_image(1, "aaa"), _image(2, "bbb"), _image(3, "ccc")],
        clusters=[_cluster(10, [1, 2]), _cluster(11, [3])],
        laser_labels=[_laser(1), _laser(2), _laser(3)],
        species_labels=[],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_species_preprocess_inputs_activity, 42
    )

    assert result.dive_id == 42
    assert result.clusters == [["aaa", "bbb"], ["ccc"]]
    assert result.camera_matrix == _K.tolist()
    assert result.distortion_coefficients == _D.tolist()


@pytest.mark.asyncio
async def test_drops_images_without_valid_laser(monkeypatch):
    """Image 2 has no valid laser (incomplete) → drop from cluster."""
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[_image(1, "aaa"), _image(2, "bbb")],
        clusters=[_cluster(10, [1, 2])],
        laser_labels=[_laser(1), _laser(2, completed=False)],
        species_labels=[],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_species_preprocess_inputs_activity, 42
    )

    assert result.clusters == [["aaa"]]


@pytest.mark.asyncio
async def test_drops_images_with_non_sentinel_species_label(monkeypatch):
    """Image 1 has an incomplete species label in a real project → already
    populated, must not re-import. Image 2 has none → keep."""
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[_image(1, "aaa"), _image(2, "bbb")],
        clusters=[_cluster(10, [1, 2])],
        laser_labels=[_laser(1), _laser(2)],
        species_labels=[_species(1, project_id=70)],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_species_preprocess_inputs_activity, 42
    )

    assert result.clusters == [["bbb"]]


@pytest.mark.asyncio
async def test_keeps_images_with_only_null_project_species_sentinels(monkeypatch):
    """NULL-project species rows are legacy sentinels — must not drop
    the image (consistent with the API cohort selector)."""
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[_image(1, "aaa")],
        clusters=[_cluster(10, [1])],
        laser_labels=[_laser(1)],
        species_labels=[_species(1, project_id=None)],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_species_preprocess_inputs_activity, 42
    )

    assert result.clusters == [["aaa"]]


@pytest.mark.asyncio
async def test_drops_clusters_that_become_empty_after_filter(monkeypatch):
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[_image(1, "aaa"), _image(2, "bbb")],
        clusters=[_cluster(10, [1]), _cluster(11, [2])],
        laser_labels=[_laser(1), _laser(2, completed=False)],
        species_labels=[],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_species_preprocess_inputs_activity, 42
    )

    assert result.clusters == [["aaa"]]


@pytest.mark.asyncio
async def test_drops_image_ids_that_no_longer_have_image_rows(monkeypatch):
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[_image(1, "aaa")],
        clusters=[_cluster(10, [1, 99])],
        laser_labels=[_laser(1), _laser(99)],
        species_labels=[],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_species_preprocess_inputs_activity, 42
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
            sut.resolve_species_preprocess_inputs_activity, 42
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
            sut.resolve_species_preprocess_inputs_activity, 42
        )


@pytest.mark.asyncio
async def test_raises_when_no_intrinsics(monkeypatch):
    fs = _make_fs(dive=_dive(), intrinsics=None, images=[], clusters=[])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    with pytest.raises(ValueError, match="no intrinsics"):
        await ActivityEnvironment().run(
            sut.resolve_species_preprocess_inputs_activity, 42
        )


# ---------- selector ↔ resolver consistency ----------
#
# The API SQL selector
# (`select_next_for_species_preprocessing`, in
# `services/fishsense-api/src/fishsense_api/controllers/dive_controller.py`)
# decides "does this dive need work?" The resolver decides "what
# work for this dive?" If they disagree, the parent workflow logs
# "0 images" and exits silently — labelers see nothing happen.
#
# These tests construct fixtures that mirror the rows of the API
# selector's truth table (see
# `services/fishsense-api/tests/test_select_next_dive_endpoints.py`,
# `test_species_preprocessing_*` block). For each row where the
# selector picks the dive, the resolver must produce ≥1 image
# checksum; for each row where the selector skips the dive, the
# resolver must either return zero clusters or its callers must not
# invoke it (parent catches selector=None before the resolver runs).
# We assert the resolver-side behavior here.


@pytest.mark.asyncio
async def test_selector_picks_dive__resolver_emits_at_least_one_image(monkeypatch):
    """SQL pair: `test_species_preprocessing_requires_prediction_cluster_and_valid_laser`
    (positive case dive 3). One PREDICTION cluster, one valid laser,
    no species row → resolver emits the image's checksum."""
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[_image(31, "ccc")],
        clusters=[_cluster(10, [31])],
        laser_labels=[_laser(31)],
        species_labels=[],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_species_preprocess_inputs_activity, 42
    )

    assert result.clusters == [["ccc"]]


@pytest.mark.asyncio
async def test_selector_skips_dive__resolver_returns_empty(monkeypatch):
    """SQL pair: `test_species_preprocessing_excludes_dive_with_only_incomplete_species_labels`.
    Every laser-valid image already has a real-project species row →
    selector returns None; if the parent ever does call the resolver
    on such a dive, it must produce zero clusters (not silently leak
    already-populated images)."""
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[_image(11, "aaa"), _image(12, "bbb")],
        clusters=[_cluster(10, [11, 12])],
        laser_labels=[_laser(11), _laser(12)],
        species_labels=[
            _species(11, project_id=70),
            _species(12, project_id=70),
        ],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_species_preprocess_inputs_activity, 42
    )

    assert not result.clusters


@pytest.mark.asyncio
async def test_selector_picks_dive__resolver_only_emits_unlabeled_subset(monkeypatch):
    """SQL pair: `test_species_preprocessing_ignores_null_project_species_sentinels`.
    Image with only a NULL-project species sentinel must remain in
    the resolver's output. Image with a real-project row must drop.
    Two images in one cluster → resolver emits the unlabeled one
    only."""
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_intrinsics(),
        images=[_image(11, "aaa"), _image(12, "bbb")],
        clusters=[_cluster(10, [11, 12])],
        laser_labels=[_laser(11), _laser(12)],
        species_labels=[
            _species(11, project_id=None),  # sentinel — keep in cohort
            _species(12, project_id=70),    # real row — drop from cohort
        ],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_species_preprocess_inputs_activity, 42
    )

    assert result.clusters == [["aaa"]]
