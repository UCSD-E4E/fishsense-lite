"""Unit tests for measure_fish_activity (stage 14).

Synthetic-geometry happy path posts a Measurement whose length matches
the constructed fish to within 1 mm. The math layer (`WorldPointHandler`)
is independently exercised by
`tests/test_compute_world_point_from_depth_convention.py` and
`tests/test_stage14_pipeline_sign_consistency.py`; this file focuses on
the SDK plumbing + result counters.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_api_sdk.models.data_source import DataSource
from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.dive_frame_cluster import DiveFrameCluster
from fishsense_api_sdk.models.fish import Fish
from fishsense_api_sdk.models.headtail_label import HeadTailLabel
from fishsense_api_sdk.models.laser_extrinsics import LaserExtrinsics
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_sdk.models.measurement import Measurement
from fishsense_api_sdk.models.species import Species
from fishsense_api_sdk.models.species_label import SpeciesLabel
from fishsense_data_processing_workflow_worker.activities import (
    measure_fish_activity as sut,
)


CAMERA_MATRIX = np.array(
    [
        [3000.0, 0.0, 2048.0],
        [0.0, 3000.0, 1536.0],
        [0.0, 0.0, 1.0],
    ]
)


def _project(point_camera: np.ndarray) -> tuple[float, float]:
    p = CAMERA_MATRIX @ point_camera
    return float(p[0] / p[2]), float(p[1] / p[2])


def _camera_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_matrix=CAMERA_MATRIX,
        distortion_coefficients=np.zeros(5),
        camera_id=1,
    )


def _dive(dive_id: int = 42) -> Dive:
    return Dive(
        id=dive_id,
        name=f"dive-{dive_id}",
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority="HIGH",
        flip_dive_slate=False,
        camera_id=1,
        dive_slate_id=7,
    )


def _species_label(
    image_id: int,
    content: str | None = "Stuff, Lingcod (Ophiodon elongatus)",
) -> SpeciesLabel:
    return SpeciesLabel(
        id=image_id * 11,
        label_studio_task_id=image_id * 13,
        label_studio_project_id=70,
        image_url=None,
        updated_at=None,
        completed=True,
        grouping="Not part of current group",
        top_three_photos_of_group=True,
        slate_upside_down=False,
        laser_x=None,
        laser_y=None,
        laser_label=None,
        content_of_image=content,
        fish_measurable_category=None,
        fish_angle_category=None,
        fish_curved_category=None,
        label_studio_json=None,
        image_id=image_id,
        user_id=None,
    )


def _laser_label(image_id: int, x: float, y: float) -> LaserLabel:
    return LaserLabel(
        id=image_id * 17,
        label_studio_task_id=image_id * 19,
        label_studio_project_id=73,
        x=x, y=y, label="laser",
        updated_at=None, superseded=False, completed=True,
        label_studio_json=None,
        image_id=image_id, user_id=None,
    )


def _headtail_label(
    image_id: int, head_xy: tuple[float, float], tail_xy: tuple[float, float]
) -> HeadTailLabel:
    return HeadTailLabel(
        id=image_id * 23,
        label_studio_task_id=image_id * 29,
        label_studio_project_id=71,
        head_x=head_xy[0], head_y=head_xy[1],
        tail_x=tail_xy[0], tail_y=tail_xy[1],
        updated_at=None, superseded=False, completed=True,
        label_studio_json=None,
        image_id=image_id, user_id=None,
    )


def _cluster(
    image_ids: list[int],
    cluster_id: int = 1,
    fish_id: int | None = None,
) -> DiveFrameCluster:
    return DiveFrameCluster(
        id=cluster_id,
        image_ids=image_ids,
        data_source=DataSource.LABEL_STUDIO,
        updated_at=None,
        dive_id=42,
        fish_id=fish_id,
    )


def _laser_extrinsics() -> LaserExtrinsics:
    # Off-axis laser: origin offset in -x, axis tilted slightly in -y.
    return LaserExtrinsics(
        laser_position=np.array([-0.03, -0.10, 0.0]),
        laser_axis=np.array([0.0, -0.02, 1.0]) / np.linalg.norm(np.array([0.0, -0.02, 1.0])),
        dive_id=42,
        camera_id=1,
        id=11,
    )


def _build_observation(
    laser_extrinsics: LaserExtrinsics,
    head_world: np.ndarray,
    tail_world: np.ndarray,
    target_depth: float,
):
    """Project the head + tail + the laser's hit at `target_depth` into
    pixels with CAMERA_MATRIX. Head/tail must lie on the plane Z=target_depth
    so the depth-from-laser back-projection recovers them exactly."""
    o = np.asarray(laser_extrinsics.laser_position, dtype=float)
    a = np.asarray(laser_extrinsics.laser_axis, dtype=float)
    t = (target_depth - o[2]) / a[2]
    laser_hit = o + t * a
    return _project(head_world), _project(tail_world), _project(laser_hit)


def _make_fs(  # pylint: disable=too-many-arguments
    *,
    dive: Dive,
    intrinsics: CameraIntrinsics | None,
    laser_extrinsics: LaserExtrinsics | None,
    species_labels: List[SpeciesLabel],
    laser_labels: dict[int, LaserLabel | None],
    headtail_labels: dict[int, HeadTailLabel | None],
    clusters: List[DiveFrameCluster],
    species_lookup: dict[str, Species] | None = None,
    new_species_id: int = 500,
    new_fish_id: int = 700,
):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(return_value=dive)
    fs.dives.get_laser_extrinsics = AsyncMock(return_value=laser_extrinsics)

    fs.cameras = MagicMock()
    fs.cameras.get_intrinsics = AsyncMock(return_value=intrinsics)

    fs.labels = MagicMock()
    fs.labels.get_species_labels = AsyncMock(return_value=species_labels)
    fs.labels.get_laser_label = AsyncMock(
        side_effect=lambda image_id=None, **_: laser_labels.get(image_id)
    )
    fs.labels.get_headtail_label = AsyncMock(
        side_effect=lambda image_id=None, **_: headtail_labels.get(image_id)
    )

    fs.images = MagicMock()
    fs.images.get_clusters = AsyncMock(return_value=clusters)
    fs.images.put_cluster = AsyncMock(return_value=clusters[0].id if clusters else None)

    fs.fish = MagicMock()
    species_lookup = species_lookup or {}
    fs.fish.get_species_by_scientific_name = AsyncMock(
        side_effect=species_lookup.get
    )
    fs.fish.post_species = AsyncMock(return_value=new_species_id)
    fs.fish.get = AsyncMock(return_value=None)
    fs.fish.post = AsyncMock(return_value=new_fish_id)
    fs.fish.post_measurement = AsyncMock(return_value=None)
    return fs


def test_parse_species_names_ok():
    assert sut._parse_species_names(  # pylint: disable=protected-access
        "Stuff, Lingcod (Ophiodon elongatus)"
    ) == ("Lingcod", "Ophiodon elongatus")


def test_parse_species_names_rejects_missing_parens():
    assert sut._parse_species_names("just text no parens") is None  # pylint: disable=protected-access


def test_parse_species_names_rejects_empty():
    assert sut._parse_species_names(None) is None  # pylint: disable=protected-access
    assert sut._parse_species_names("") is None  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_raises_when_dive_missing(monkeypatch):
    fs = _make_fs(
        dive=None,
        intrinsics=_camera_intrinsics(),
        laser_extrinsics=_laser_extrinsics(),
        species_labels=[], laser_labels={}, headtail_labels={}, clusters=[],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    with pytest.raises(ValueError, match="not found"):
        await ActivityEnvironment().run(sut.measure_fish_activity, 42)


@pytest.mark.asyncio
async def test_raises_when_laser_extrinsics_missing(monkeypatch):
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_camera_intrinsics(),
        laser_extrinsics=None,
        species_labels=[], laser_labels={}, headtail_labels={}, clusters=[],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    with pytest.raises(ValueError, match="laser_extrinsics"):
        await ActivityEnvironment().run(sut.measure_fish_activity, 42)


@pytest.mark.asyncio
async def test_skips_species_label_with_no_cluster(monkeypatch):
    image_id = 100
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_camera_intrinsics(),
        laser_extrinsics=_laser_extrinsics(),
        species_labels=[_species_label(image_id)],
        laser_labels={image_id: _laser_label(image_id, 2000.0, 1500.0)},
        headtail_labels={image_id: _headtail_label(image_id, (1900.0, 1500.0), (2100.0, 1500.0))},
        clusters=[],  # <-- no clusters at all
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(sut.measure_fish_activity, 42)
    assert result.measured == 0
    assert result.missing_cluster == 1
    fs.fish.post_measurement.assert_not_called()
    fs.fish.post.assert_not_called()


@pytest.mark.asyncio
async def test_skips_when_laser_or_headtail_label_missing(monkeypatch):
    image_id = 100
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_camera_intrinsics(),
        laser_extrinsics=_laser_extrinsics(),
        species_labels=[_species_label(image_id)],
        laser_labels={image_id: None},  # <-- missing
        headtail_labels={image_id: None},
        clusters=[_cluster([image_id])],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(sut.measure_fish_activity, 42)
    assert result.measured == 0
    assert result.missing_laser_or_headtail == 1
    fs.fish.post_measurement.assert_not_called()


@pytest.mark.asyncio
async def test_filters_out_non_top_three_labels(monkeypatch):
    img_a, img_b = 100, 101
    sp_a = _species_label(img_a)
    sp_b = _species_label(img_b)
    sp_b.top_three_photos_of_group = False  # excluded
    fs = _make_fs(
        dive=_dive(),
        intrinsics=_camera_intrinsics(),
        laser_extrinsics=_laser_extrinsics(),
        species_labels=[sp_a, sp_b],
        laser_labels={},
        headtail_labels={},
        clusters=[_cluster([img_a, img_b])],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(sut.measure_fish_activity, 42)
    # Only `img_a` was attempted; missing laser/headtail counter == 1.
    assert result.missing_laser_or_headtail == 1
    assert fs.labels.get_laser_label.await_count == 1


@pytest.mark.asyncio
async def test_measures_one_fish_end_to_end(monkeypatch):
    image_id = 100
    le = _laser_extrinsics()
    intrinsics = _camera_intrinsics()
    target_depth = 1.20  # meters

    head_world = np.array([-0.10, 0.00, target_depth])
    tail_world = np.array([0.20, 0.00, target_depth])
    expected_length = float(np.linalg.norm(head_world - tail_world))

    head_pix, tail_pix, laser_pix = _build_observation(
        le, head_world, tail_world, target_depth
    )

    fs = _make_fs(
        dive=_dive(),
        intrinsics=intrinsics,
        laser_extrinsics=le,
        species_labels=[_species_label(image_id)],
        laser_labels={image_id: _laser_label(image_id, *laser_pix)},
        headtail_labels={image_id: _headtail_label(image_id, head_pix, tail_pix)},
        clusters=[_cluster([image_id], fish_id=None)],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(sut.measure_fish_activity, 42)

    assert result.measured == 1
    assert result.dropped_nan == 0
    assert result.missing_laser_or_headtail == 0
    assert result.missing_cluster == 0

    # Species was created (lookup returned None).
    fs.fish.post_species.assert_awaited_once()
    posted_species = fs.fish.post_species.call_args.args[0]
    assert posted_species.scientific_name == "Ophiodon elongatus"
    assert posted_species.common_name == "Lingcod"

    # Fish was created (cluster.fish_id was None) and the cluster rebound.
    fs.fish.post.assert_awaited_once()
    fs.images.put_cluster.assert_awaited_once()

    # Measurement length is within 1 mm of ground truth.
    fs.fish.post_measurement.assert_awaited_once()
    fish_id, measurement = fs.fish.post_measurement.call_args.args
    assert isinstance(measurement, Measurement)
    assert measurement.image_id == image_id
    assert measurement.fish_id == fish_id
    assert abs(measurement.length_m - expected_length) < 1e-3


@pytest.mark.asyncio
async def test_existing_species_and_fish_are_reused(monkeypatch):
    image_id = 100
    le = _laser_extrinsics()
    intrinsics = _camera_intrinsics()
    target_depth = 1.0
    head_world = np.array([-0.1, 0.0, target_depth])
    tail_world = np.array([0.1, 0.0, target_depth])
    head_pix, tail_pix, laser_pix = _build_observation(
        le, head_world, tail_world, target_depth
    )

    existing_species = Species(id=42, common_name="Lingcod", scientific_name="Ophiodon elongatus")
    existing_fish = Fish(id=88, species_id=existing_species.id)

    fs = _make_fs(
        dive=_dive(),
        intrinsics=intrinsics,
        laser_extrinsics=le,
        species_labels=[_species_label(image_id)],
        laser_labels={image_id: _laser_label(image_id, *laser_pix)},
        headtail_labels={image_id: _headtail_label(image_id, head_pix, tail_pix)},
        clusters=[_cluster([image_id], fish_id=existing_fish.id)],
        species_lookup={"Ophiodon elongatus": existing_species},
    )
    fs.fish.get = AsyncMock(return_value=existing_fish)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(sut.measure_fish_activity, 42)

    assert result.measured == 1
    fs.fish.post_species.assert_not_called()
    fs.fish.post.assert_not_called()
    # Cluster already points at the right fish — no re-PUT.
    fs.images.put_cluster.assert_not_called()
    fish_id_called, _ = fs.fish.post_measurement.call_args.args
    assert fish_id_called == existing_fish.id
