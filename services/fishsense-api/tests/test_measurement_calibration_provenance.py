# pylint: disable=C0121
"""Which calibration a `Measurement` was computed with.

A fish length is only meaningful relative to the `LaserExtrinsics` that
produced the depth behind it, and calibrations do get replaced — the
2026-08-11 slate panel-offset fix recalibrated 6 of the 8 dives that already
had measurements. Before this column there was no way to ask "is this length
still current?", so the choice was to rewrite every measurement on every run
or to leave stale ones in place forever. Stamping the provenance turns it
into an ordinary drainable cohort: stage 14 skips an image only when its
measurement was computed with the calibration that would be used *today*, and
re-measures it otherwise.

Legacy rows carry NULL and therefore read as stale exactly once — that is the
length half of the depth backfill, and it converges without a separate script.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_shared import taxonomy


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


def _dive(dive_id: int, *, calibration_dive_id: int | None = None):
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    return Dive(
        id=dive_id,
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority=Priority.HIGH,
        calibration_dive_id=calibration_dive_id,
    )


def _extrinsics(extrinsics_id: int, dive_id: int):
    from fishsense_api.models.laser_extrinsics import LaserExtrinsics  # pylint: disable=import-outside-toplevel

    return LaserExtrinsics(
        id=extrinsics_id,
        laser_position=[0.1, 0.0, 0.0],
        laser_axis=[0.0, 0.0, 1.0],
        dive_id=dive_id,
        camera_id=1,
    )


def _measurable_image(session, image_id: int, dive_id: int):
    """A frame stage 14 would attempt: a fish-model top-three species label
    plus valid laser and head/tail labels. Models need no LABEL_STUDIO
    cluster, which keeps the fixture to the parts this test is about."""
    from fishsense_api.models.head_tail_label import HeadTailLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(
        Image(
            id=image_id,
            path=f"/dev/null/img-{image_id}",
            taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
            checksum=f"{image_id:032d}",
            is_canonical=True,
            dive_id=dive_id,
        )
    )
    session.add(
        LaserLabel(image_id=image_id, completed=True, superseded=False, x=1.0, y=2.0)
    )
    session.add(
        HeadTailLabel(
            image_id=image_id,
            completed=True,
            superseded=False,
            head_x=1.0,
            head_y=2.0,
            tail_x=3.0,
            tail_y=4.0,
        )
    )
    session.add(
        SpeciesLabel(
            image_id=image_id,
            top_three_photos_of_group=True,
            completed=True,
            superseded=False,
            label_studio_project_id=70,
            content_of_image=f"{taxonomy.FISH_MODEL_PREFIX} Grouper",
        )
    )


def _measurement(image_id: int, *, laser_extrinsics_id=None, fish_id: int = 100):
    from fishsense_api.models.measurement import Measurement  # pylint: disable=import-outside-toplevel

    return Measurement(
        image_id=image_id,
        fish_id=fish_id,
        length_m=0.36,
        laser_extrinsics_id=laser_extrinsics_id,
    )


async def test_post_measurement_persists_the_calibration_it_used(session):
    from fishsense_api.controllers.fish_controller import (  # pylint: disable=import-outside-toplevel
        post_measurement,
    )
    from fishsense_api.models.measurement import Measurement  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1), _extrinsics(51, 1)])
    _measurable_image(session, 11, 1)
    await session.flush()

    measurement_id = await post_measurement(
        100, _measurement(11, laser_extrinsics_id=51), session=session
    )

    row = await session.get(Measurement, measurement_id)
    assert row.laser_extrinsics_id == 51


async def test_post_measurement_updates_provenance_on_re_measure(session):
    """The upsert keys on `(image_id, fish_id)`, so a re-measure after a
    recalibration must move the provenance forward with the length — a row
    still claiming the old calibration would be re-selected forever."""
    from fishsense_api.controllers.fish_controller import (  # pylint: disable=import-outside-toplevel
        post_measurement,
    )
    from fishsense_api.models.measurement import Measurement  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1), _extrinsics(51, 1)])
    _measurable_image(session, 11, 1)
    await session.flush()

    first = await post_measurement(
        100, _measurement(11, laser_extrinsics_id=50), session=session
    )
    second = await post_measurement(
        100, _measurement(11, laser_extrinsics_id=51), session=session
    )

    assert first == second
    row = await session.get(Measurement, second)
    assert row.laser_extrinsics_id == 51


async def test_cohort_skips_a_dive_measured_with_the_current_calibration(session):
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )

    session.add_all([_dive(1), _extrinsics(51, 1)])
    _measurable_image(session, 11, 1)
    session.add(_measurement(11, laser_extrinsics_id=51))
    await session.flush()

    assert await select_next_for_measure_fish(session=session) is None


async def test_cohort_repicks_a_dive_after_recalibration(session):
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )

    session.add_all([_dive(1), _extrinsics(51, 1)])
    _measurable_image(session, 11, 1)
    session.add(_measurement(11, laser_extrinsics_id=50))
    await session.flush()

    assert await select_next_for_measure_fish(session=session) == 1


async def test_cohort_repicks_a_dive_whose_measurements_predate_provenance(session):
    """Every measurement in prod today carries NULL here. They re-enter the
    cohort once, get recomputed under the current calibration, and drain."""
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )

    session.add_all([_dive(1), _extrinsics(51, 1)])
    _measurable_image(session, 11, 1)
    session.add(_measurement(11, laser_extrinsics_id=None))
    await session.flush()

    assert await select_next_for_measure_fish(session=session) == 1


async def test_cohort_resolves_provenance_through_a_borrowed_calibration(session):
    """A fish-only dive is measured with the sibling's extrinsics row, so
    that is the id its measurements must carry — not its own (it has none)."""
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )

    session.add_all([_dive(1, calibration_dive_id=2), _dive(2), _extrinsics(51, 2)])
    _measurable_image(session, 11, 1)
    session.add(_measurement(11, laser_extrinsics_id=51))
    await session.flush()

    assert await select_next_for_measure_fish(session=session) is None
