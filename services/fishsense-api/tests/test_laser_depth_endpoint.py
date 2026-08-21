# pylint: disable=C0121
#   `== True` is SQLAlchemy expression building, not a sloppy truth test.
"""Per-image laser depth: store, read, and the cohort that produces it.

`LaserDepth` answers "how far away was the laser dot in this frame?" for
*every* image carrying a valid laser label — not just the measurable ones
stage 14 visits. It is derived data (laser label x/y + the dive's resolved
`LaserExtrinsics` + the camera's intrinsics, through
`WorldPointHandler.compute_world_point_from_laser`), so the row records
**which** label and **which** calibration produced it. That provenance is
what makes the value falsifiable: a relabel or a recalibration invalidates
the depth, and the cohort selector below re-picks the dive on exactly that
condition rather than treating a stale number as done.

FK-less in-memory sqlite, same as the other controller tests — these pin
query composition and write semantics, not referential integrity.
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


def _dive(dive_id: int, *, priority=None, calibration_dive_id: int | None = None):
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    return Dive(
        id=dive_id,
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority=Priority.HIGH if priority is None else priority,
        calibration_dive_id=calibration_dive_id,
    )


def _image(image_id: int, dive_id: int, *, is_canonical: bool = True):
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    return Image(
        id=image_id,
        path=f"/dev/null/img-{image_id}",
        taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        checksum=f"{image_id:032d}",
        dive_id=dive_id,
        is_canonical=is_canonical,
    )


def _laser_label(label_id: int, image_id: int, *, completed=True, superseded=False):
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    return LaserLabel(
        id=label_id,
        x=100.0,
        y=200.0,
        completed=completed,
        superseded=superseded,
        image_id=image_id,
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


def _depth(
    *,
    depth_m=2.0,
    range_m=2.01,
    residual_m=0.0,
    laser_label_id=None,
    laser_extrinsics_id=None,
):
    from fishsense_api.models.laser_depth import LaserDepth  # pylint: disable=import-outside-toplevel

    return LaserDepth(
        depth_m=depth_m,
        range_m=range_m,
        residual_m=residual_m,
        laser_label_id=laser_label_id,
        laser_extrinsics_id=laser_extrinsics_id,
    )


# ── store + read ──────────────────────────────────────────────────────


async def test_put_creates_a_depth(session):
    from fishsense_api.controllers.laser_depth_controller import (  # pylint: disable=import-outside-toplevel
        put_laser_depth,
    )
    from fishsense_api.models.laser_depth import LaserDepth  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    depth_id = await put_laser_depth(
        11,
        _depth(depth_m=2.14, range_m=2.19, laser_label_id=5, laser_extrinsics_id=7),
        session=session,
    )

    row = await session.get(LaserDepth, depth_id)
    assert row.image_id == 11
    assert (row.depth_m, row.range_m) == (2.14, 2.19)
    assert (row.laser_label_id, row.laser_extrinsics_id) == (5, 7)


async def test_put_records_the_triangulation_residual(session):
    """How far apart the camera ray and the laser ray actually passed.

    Stored, not gated on: it is a *necessary but not sufficient* check —
    blind to error along the laser's epipolar line, zero for two rays meeting
    at the camera centre, and metric, so no fixed threshold is right at every
    depth. Recording it first means a threshold can be chosen from the real
    distribution instead of guessed.
    """
    from fishsense_api.controllers.laser_depth_controller import (  # pylint: disable=import-outside-toplevel
        put_laser_depth,
    )
    from fishsense_api.models.laser_depth import LaserDepth  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    depth_id = await put_laser_depth(11, _depth(residual_m=0.0031), session=session)

    row = await session.get(LaserDepth, depth_id)
    assert row.residual_m == pytest.approx(0.0031)


async def test_put_upserts_on_image_id(session):
    """Recompute overwrites. `merge` on `id=None` would always INSERT and
    violate `uq_laser_depth_image` — the same natural-key trap that
    duplicated measurements and 500'd the label writeback."""
    from fishsense_api.controllers.laser_depth_controller import (  # pylint: disable=import-outside-toplevel
        put_laser_depth,
    )
    from fishsense_api.models.laser_depth import LaserDepth  # pylint: disable=import-outside-toplevel
    from sqlmodel import select  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    first = await put_laser_depth(11, _depth(depth_m=2.0), session=session)
    second = await put_laser_depth(11, _depth(depth_m=3.5), session=session)

    assert first == second
    rows = (await session.exec(select(LaserDepth))).all()
    assert len(rows) == 1
    assert rows[0].depth_m == 3.5


async def test_get_returns_404_when_the_image_has_no_depth(session):
    from fastapi import HTTPException  # pylint: disable=import-outside-toplevel

    from fishsense_api.controllers.laser_depth_controller import (  # pylint: disable=import-outside-toplevel
        get_laser_depth,
    )

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    with pytest.raises(HTTPException) as excinfo:
        await get_laser_depth(11, session=session)
    assert excinfo.value.status_code == 404


async def test_get_returns_the_stored_depth(session):
    from fishsense_api.controllers.laser_depth_controller import (  # pylint: disable=import-outside-toplevel
        get_laser_depth,
        put_laser_depth,
    )

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()
    await put_laser_depth(11, _depth(depth_m=1.75, range_m=1.80), session=session)

    row = await get_laser_depth(11, session=session)
    assert (row.depth_m, row.range_m) == (1.75, 1.80)


async def test_get_for_dive_returns_only_that_dives_rows(session):
    """Empty list, not 404, when a dive has none — "no depth yet" is a
    normal pipeline state, not an error, and the caller iterates."""
    from fishsense_api.controllers.laser_depth_controller import (  # pylint: disable=import-outside-toplevel
        get_laser_depths_for_dive,
        put_laser_depth,
    )

    session.add_all([_dive(1), _dive(2), _image(11, 1), _image(12, 1), _image(21, 2)])
    await session.flush()
    await put_laser_depth(11, _depth(depth_m=1.0), session=session)
    await put_laser_depth(12, _depth(depth_m=2.0), session=session)
    await put_laser_depth(21, _depth(depth_m=9.0), session=session)

    rows = await get_laser_depths_for_dive(1, session=session)
    assert sorted(r.depth_m for r in rows) == [1.0, 2.0]
    assert await get_laser_depths_for_dive(3, session=session) == []


# ── the cohort ────────────────────────────────────────────────────────


async def test_selector_picks_a_dive_whose_laser_images_have_no_depth(session):
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_depth,
    )

    session.add_all(
        [_dive(1), _image(11, 1), _laser_label(101, 11), _extrinsics(51, 1)]
    )
    await session.flush()

    assert await select_next_for_laser_depth(session=session) == 1


async def test_selector_requires_calibration(session):
    """No extrinsics, no depth: `compute_world_point_from_laser` needs the
    laser's position and axis. Such a dive is stage 13's problem, not this
    stage's, and must not sit in the cohort forever."""
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_depth,
    )

    session.add_all([_dive(1), _image(11, 1), _laser_label(101, 11)])
    await session.flush()

    assert await select_next_for_laser_depth(session=session) is None


async def test_selector_accepts_borrowed_calibration(session):
    """Mirrors `get_laser_extrinsics_for_dive`: a fish-only dive borrows a
    sibling slate dive's rig calibration via `Dive.calibration_dive_id`."""
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_depth,
    )

    session.add_all(
        [
            _dive(1, calibration_dive_id=2),
            _dive(2),
            _image(11, 1),
            _laser_label(101, 11),
            _extrinsics(51, 2),
        ]
    )
    await session.flush()

    assert await select_next_for_laser_depth(session=session) == 1


async def test_selector_skips_a_dive_whose_depths_are_current(session):
    """Drains. The depth row names the label and the calibration it came
    from; when both still match, there is nothing to recompute."""
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_depth,
    )
    from fishsense_api.controllers.laser_depth_controller import (  # pylint: disable=import-outside-toplevel
        put_laser_depth,
    )

    session.add_all(
        [_dive(1), _image(11, 1), _laser_label(101, 11), _extrinsics(51, 1)]
    )
    await session.flush()
    await put_laser_depth(
        11, _depth(laser_label_id=101, laser_extrinsics_id=51), session=session
    )

    assert await select_next_for_laser_depth(session=session) is None


async def test_selector_repicks_a_dive_after_recalibration(session):
    """The 2026-08-11 slate panel-offset fix recalibrated dives whose depths
    had already been computed. A depth carrying the superseded extrinsics id
    is stale, and staleness is the whole reason the provenance columns exist."""
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_depth,
    )
    from fishsense_api.controllers.laser_depth_controller import (  # pylint: disable=import-outside-toplevel
        put_laser_depth,
    )

    session.add_all(
        [_dive(1), _image(11, 1), _laser_label(101, 11), _extrinsics(51, 1)]
    )
    await session.flush()
    await put_laser_depth(
        11, _depth(laser_label_id=101, laser_extrinsics_id=50), session=session
    )

    assert await select_next_for_laser_depth(session=session) == 1


async def test_selector_repicks_a_dive_after_the_laser_label_changes(session):
    """A superseded-then-replaced laser label moves the dot, which moves the
    depth. Keyed on the label id, so the replacement re-enters the cohort."""
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_depth,
    )
    from fishsense_api.controllers.laser_depth_controller import (  # pylint: disable=import-outside-toplevel
        put_laser_depth,
    )

    session.add_all(
        [
            _dive(1),
            _image(11, 1),
            _laser_label(101, 11, superseded=True),
            _laser_label(102, 11),
            _extrinsics(51, 1),
        ]
    )
    await session.flush()
    await put_laser_depth(
        11, _depth(laser_label_id=101, laser_extrinsics_id=51), session=session
    )

    assert await select_next_for_laser_depth(session=session) == 1


async def test_selector_ignores_images_without_a_valid_laser(session):
    """Incomplete, superseded, or coordinate-less labels are not a laser
    fix — the same `_valid_laser_conditions` gate stages 1/2/5.1/14 use."""
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_depth,
    )

    session.add_all(
        [
            _dive(1),
            _image(11, 1),
            _laser_label(101, 11, completed=False),
            _image(12, 1),
            _laser_label(102, 12, superseded=True),
            _extrinsics(51, 1),
        ]
    )
    await session.flush()

    assert await select_next_for_laser_depth(session=session) is None


async def test_selector_ignores_non_canonical_images(session):
    """The prod dive 60 wedge: a duplicate dive's frames never drain because
    the pipeline declines to work on them. Every selector filters canonical."""
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_depth,
    )

    session.add_all(
        [
            _dive(1),
            _image(11, 1, is_canonical=False),
            _laser_label(101, 11),
            _extrinsics(51, 1),
        ]
    )
    await session.flush()

    assert await select_next_for_laser_depth(session=session) is None


async def test_selector_skips_low_priority_dives(session):
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_depth,
    )
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    session.add_all(
        [
            _dive(1, priority=Priority.LOW),
            _image(11, 1),
            _laser_label(101, 11),
            _extrinsics(51, 1),
        ]
    )
    await session.flush()

    assert await select_next_for_laser_depth(session=session) is None


# ── duplicate valid labels ────────────────────────────────────────────
#
# `LaserDepth` is one row per image, but an image can carry more than one
# *valid* laser label: 461 images across 8 prod dives do, almost all of them
# duplicate labels of the same dot (position spread 0.000 px). Keying the
# cohort on "is there a depth for THIS label" therefore never went false for
# those images — the row records one label id and the other never matches.
#
# Prod dive 279 was the proof: 44 eligible images, 44 depths written, and the
# selector still offered it on 27 consecutive runs. That is the same
# never-drains shape as the stage-14 `Fish Model,` empty leaf, and it blocks
# every higher-id dive behind it.
#
# The predicate now asks the question the table actually answers: does this
# *image* have a depth derived from a still-valid label and the current
# calibration?


async def test_selector_drains_an_image_with_two_valid_laser_labels(session):
    """Both labels are valid and the depth names one of them — that is a
    complete answer for the image, so the dive must drop out."""
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_depth,
    )
    from fishsense_api.controllers.laser_depth_controller import (  # pylint: disable=import-outside-toplevel
        put_laser_depth,
    )

    session.add_all(
        [
            _dive(1),
            _image(11, 1),
            _laser_label(101, 11),
            _laser_label(102, 11),  # duplicate label of the same dot
            _extrinsics(51, 1),
        ]
    )
    await session.flush()
    await put_laser_depth(
        11, _depth(laser_label_id=101, laser_extrinsics_id=51), session=session
    )

    assert await select_next_for_laser_depth(session=session) is None


async def test_selector_still_repicks_when_the_recorded_label_went_superseded(session):
    """The self-healing property must survive the fix: a depth whose label is
    no longer valid is stale, even though the image has another valid one."""
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_depth,
    )
    from fishsense_api.controllers.laser_depth_controller import (  # pylint: disable=import-outside-toplevel
        put_laser_depth,
    )

    session.add_all(
        [
            _dive(1),
            _image(11, 1),
            _laser_label(101, 11, superseded=True),
            _laser_label(102, 11),
            _extrinsics(51, 1),
        ]
    )
    await session.flush()
    await put_laser_depth(
        11, _depth(laser_label_id=101, laser_extrinsics_id=51), session=session
    )

    assert await select_next_for_laser_depth(session=session) == 1
