"""Tests for the per-stage `GET /api/v1/dives/select-next/...` endpoints.

These collapse the api-workflow-worker's previously O(N) selector
activities (one HTTP call per HIGH-priority dive) into a single
SELECT … LIMIT 1. The cohort definitions live in CLAUDE.md; this file
pins them down at the SQL layer.

Uses the same FK-less in-memory sqlite fixture as
test_get_clusters_data_source_filter.py — we're testing query
composition, not referential integrity.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession


@pytest.fixture
async def session():
    import fishsense_api.database  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _dive(dive_id: int, *, priority="HIGH", dive_slate_id=None):
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    return Dive(
        id=dive_id,
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority=Priority[priority],
        dive_slate_id=dive_slate_id,
    )


def _image(image_id: int, dive_id: int):
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    return Image(
        id=image_id,
        path=f"/dev/null/img-{image_id}",
        taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        checksum=f"img-{image_id:032d}"[:32],
        dive_id=dive_id,
    )


# ---------- stage 0.1: laser-preprocessing ----------


async def test_laser_preprocessing_picks_lowest_high_priority_without_extrinsics(
    session,
):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )
    from fishsense_api.models.laser_extrinsics import (  # pylint: disable=import-outside-toplevel
        LaserExtrinsics,
    )

    session.add_all(
        [
            _dive(2, priority="HIGH"),
            _dive(1, priority="HIGH"),
            _dive(3, priority="LOW"),
        ]
    )
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) == 1


async def test_laser_preprocessing_skips_dives_with_extrinsics(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )
    from fishsense_api.models.laser_extrinsics import (  # pylint: disable=import-outside-toplevel
        LaserExtrinsics,
    )

    session.add_all([_dive(1), _dive(2), _dive(3)])
    await session.flush()
    session.add_all(
        [
            LaserExtrinsics(dive_id=1, camera_id=1),
            LaserExtrinsics(dive_id=2, camera_id=1),
        ]
    )
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) == 3


async def test_laser_preprocessing_returns_none_when_all_calibrated(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )
    from fishsense_api.models.laser_extrinsics import (  # pylint: disable=import-outside-toplevel
        LaserExtrinsics,
    )

    session.add_all([_dive(1), _dive(2)])
    await session.flush()
    session.add_all(
        [LaserExtrinsics(dive_id=1, camera_id=1), LaserExtrinsics(dive_id=2, camera_id=1)]
    )
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) is None


async def test_laser_preprocessing_returns_none_with_no_high_priority(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )

    session.add_all([_dive(1, priority="LOW"), _dive(2, priority="LOW")])
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) is None


# ---------- stage 2: dive-image-preprocessing ----------


async def test_dive_image_preprocessing_requires_prediction_cluster(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_dive_image_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )

    # dive 1: no clusters -> excluded.
    # dive 2: PREDICTION cluster + image without species_label -> picked.
    session.add_all([_dive(1), _dive(2)])
    await session.flush()
    session.add_all([_image(101, 1), _image(202, 2)])
    await session.flush()
    session.add(DiveFrameCluster(dive_id=2, data_source=DataSource.PREDICTION))
    await session.flush()

    assert await select_next_for_dive_image_preprocessing(session=session) == 2


async def test_dive_image_preprocessing_skips_when_all_images_completed(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_dive_image_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    # dive 1: 1 image, all completed -> excluded.
    # dive 2: 2 images, only one completed -> picked.
    session.add_all([_dive(1), _dive(2)])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2), _image(22, 2)])
    await session.flush()
    session.add_all(
        [
            DiveFrameCluster(dive_id=1, data_source=DataSource.PREDICTION),
            DiveFrameCluster(dive_id=2, data_source=DataSource.PREDICTION),
        ]
    )
    session.add_all(
        [
            SpeciesLabel(image_id=11, completed=True),
            SpeciesLabel(image_id=21, completed=True),
            # image 22 has no species_label -> dive 2 stays in cohort.
        ]
    )
    await session.flush()

    assert await select_next_for_dive_image_preprocessing(session=session) == 2


async def test_dive_image_preprocessing_returns_none_when_only_label_studio_clusters(
    session,
):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_dive_image_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    session.add(DiveFrameCluster(dive_id=1, data_source=DataSource.LABEL_STUDIO))
    await session.flush()

    assert await select_next_for_dive_image_preprocessing(session=session) is None


# ---------- stage 5.1: headtail-preprocessing ----------


async def test_headtail_preprocessing_requires_top_three_without_completed_headtail(
    session,
):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_headtail_preprocessing,
    )
    from fishsense_api.models.head_tail_label import HeadTailLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    # dive 1: top_three=True but headtail completed -> excluded.
    # dive 2: top_three=True, no headtail -> picked.
    # dive 3: only top_three=False -> excluded.
    session.add_all([_dive(1), _dive(2), _dive(3)])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2), _image(31, 3)])
    await session.flush()
    session.add_all(
        [
            SpeciesLabel(image_id=11, top_three_photos_of_group=True),
            SpeciesLabel(image_id=21, top_three_photos_of_group=True),
            SpeciesLabel(image_id=31, top_three_photos_of_group=False),
        ]
    )
    session.add(HeadTailLabel(image_id=11, completed=True))
    await session.flush()

    assert await select_next_for_headtail_preprocessing(session=session) == 2


async def test_headtail_preprocessing_returns_none_when_no_top_three(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_headtail_preprocessing,
    )
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    session.add(SpeciesLabel(image_id=11, top_three_photos_of_group=False))
    await session.flush()

    assert await select_next_for_headtail_preprocessing(session=session) is None


# ---------- stage 9: slate-preprocessing ----------


async def test_slate_preprocessing_requires_dive_slate_id_and_marker(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        SLATE_CONTENT_MARKER,
        select_next_for_slate_preprocessing,
    )
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    # dive 1: HIGH but no dive_slate_id -> excluded.
    # dive 2: HIGH + dive_slate_id but no slate-marked species_label -> excluded.
    # dive 3: HIGH + dive_slate_id + slate-marked + no completed slate label -> picked.
    session.add_all(
        [
            _dive(1, dive_slate_id=None),
            _dive(2, dive_slate_id=99),
            _dive(3, dive_slate_id=99),
        ]
    )
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2), _image(31, 3)])
    await session.flush()
    session.add_all(
        [
            SpeciesLabel(image_id=11, content_of_image=SLATE_CONTENT_MARKER),
            SpeciesLabel(image_id=21, content_of_image="Fish"),
            SpeciesLabel(image_id=31, content_of_image=SLATE_CONTENT_MARKER),
        ]
    )
    await session.flush()

    assert await select_next_for_slate_preprocessing(session=session) == 3


async def test_slate_preprocessing_skips_when_slate_label_completed(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        SLATE_CONTENT_MARKER,
        select_next_for_slate_preprocessing,
    )
    from fishsense_api.models.dive_slate_label import DiveSlateLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1, dive_slate_id=99), _dive(2, dive_slate_id=99)])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2)])
    await session.flush()
    session.add_all(
        [
            SpeciesLabel(image_id=11, content_of_image=SLATE_CONTENT_MARKER),
            SpeciesLabel(image_id=21, content_of_image=SLATE_CONTENT_MARKER),
            DiveSlateLabel(image_id=11, completed=True),
        ]
    )
    await session.flush()

    assert await select_next_for_slate_preprocessing(session=session) == 2


# ---------- stage 13: laser-calibration ----------


async def test_laser_calibration_requires_min_completed_slate_labels(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_calibration,
    )
    from fishsense_api.models.dive_slate_label import DiveSlateLabel  # pylint: disable=import-outside-toplevel

    # dive 1: 1 completed slate label -> below threshold, excluded.
    # dive 2: 2 completed slate labels -> picked.
    # dive 3: 3 completed slate labels but already has extrinsics -> excluded.
    from fishsense_api.models.laser_extrinsics import (  # pylint: disable=import-outside-toplevel
        LaserExtrinsics,
    )

    session.add_all(
        [
            _dive(1, dive_slate_id=99),
            _dive(2, dive_slate_id=99),
            _dive(3, dive_slate_id=99),
        ]
    )
    await session.flush()
    session.add_all(
        [
            _image(11, 1),
            _image(21, 2),
            _image(22, 2),
            _image(31, 3),
            _image(32, 3),
            _image(33, 3),
        ]
    )
    await session.flush()
    session.add_all(
        [
            DiveSlateLabel(image_id=11, completed=True),
            DiveSlateLabel(image_id=21, completed=True),
            DiveSlateLabel(image_id=22, completed=True),
            DiveSlateLabel(image_id=31, completed=True),
            DiveSlateLabel(image_id=32, completed=True),
            DiveSlateLabel(image_id=33, completed=True),
            LaserExtrinsics(dive_id=3, camera_id=1),
        ]
    )
    await session.flush()

    assert await select_next_for_laser_calibration(session=session) == 2


async def test_laser_calibration_requires_dive_slate_id(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_calibration,
    )
    from fishsense_api.models.dive_slate_label import DiveSlateLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1, dive_slate_id=None))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    session.add_all(
        [
            DiveSlateLabel(image_id=11, completed=True),
            DiveSlateLabel(image_id=12, completed=True),
        ]
    )
    await session.flush()

    assert await select_next_for_laser_calibration(session=session) is None


# ---------- stage 14: measure-fish ----------


async def test_measure_fish_requires_extrinsics_and_unbound_label_studio_cluster(
    session,
):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.laser_extrinsics import (  # pylint: disable=import-outside-toplevel
        LaserExtrinsics,
    )

    # dive 1: no extrinsics -> excluded.
    # dive 2: extrinsics + LABEL_STUDIO cluster all bound -> excluded.
    # dive 3: extrinsics + unbound LABEL_STUDIO cluster -> picked.
    # dive 4: extrinsics + only PREDICTION clusters -> excluded.
    session.add_all([_dive(1), _dive(2), _dive(3), _dive(4)])
    await session.flush()
    session.add_all(
        [
            LaserExtrinsics(dive_id=2, camera_id=1),
            LaserExtrinsics(dive_id=3, camera_id=1),
            LaserExtrinsics(dive_id=4, camera_id=1),
            DiveFrameCluster(
                dive_id=2, data_source=DataSource.LABEL_STUDIO, fish_id=42
            ),
            DiveFrameCluster(
                dive_id=3, data_source=DataSource.LABEL_STUDIO, fish_id=None
            ),
            DiveFrameCluster(
                dive_id=4, data_source=DataSource.PREDICTION, fish_id=None
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_measure_fish(session=session) == 3


async def test_measure_fish_returns_none_when_all_clusters_bound(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.laser_extrinsics import (  # pylint: disable=import-outside-toplevel
        LaserExtrinsics,
    )

    session.add(_dive(1))
    await session.flush()
    session.add_all(
        [
            LaserExtrinsics(dive_id=1, camera_id=1),
            DiveFrameCluster(
                dive_id=1, data_source=DataSource.LABEL_STUDIO, fish_id=42
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_measure_fish(session=session) is None
