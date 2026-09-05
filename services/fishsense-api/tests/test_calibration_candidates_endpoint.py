"""Tests for the calibration-borrow candidate finder.

`get_calibration_candidates` returns dives whose laser-line fingerprint matches
the target's, gated on (same camera + candidate has own extrinsics + both fits
confident + line within tolerance) and ranked by line closeness. Suggest-only.
FK-less in-memory sqlite harness.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

_T0 = datetime(2024, 6, 1, tzinfo=timezone.utc)


@pytest.fixture
async def session():
    import fishsense_api.database  # noqa: F401  # pylint: disable=unused-import

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _dive(dive_id: int, camera_id: int | None, *, days: int = 0):
    from fishsense_api.models.dive import Dive
    from fishsense_api.models.priority import (
        Priority,
    )

    return Dive(
        id=dive_id,
        name=f"dive-{dive_id}",
        path=f"/dev/null/{dive_id}",
        dive_datetime=_T0 + timedelta(days=days),
        priority=Priority.HIGH,
        camera_id=camera_id,
    )


def _line(dive_id: int, angle_deg: float, c: float, *, confidence: float = 5000.0):
    """A unit-normal Hesse line rotated `angle_deg` from the x-axis normal."""
    from fishsense_api.models.dive_laser_line import (
        DiveLaserLine,
    )

    rad = math.radians(angle_deg)
    return DiveLaserLine(
        dive_id=dive_id,
        a=math.cos(rad),
        b=math.sin(rad),
        c=c,
        n_points=30,
        inlier_count=29,
        inlier_fraction=0.96,
        residual_std=0.5,
        label_noise_mad=0.4,
        line_confidence=confidence,
    )


def _extrinsics(dive_id: int, camera_id: int):
    from fishsense_api.models.laser_extrinsics import (
        LaserExtrinsics,
    )

    return LaserExtrinsics(
        dive_id=dive_id,
        camera_id=camera_id,
        laser_position=[0.0, 0.0, 0.0],
        laser_axis=[0.0, 0.0, 1.0],
        created_at=_T0,
    )


async def _candidates(session, dive_id, **kw):
    from fishsense_api.controllers.calibration_candidate_controller import (
        get_calibration_candidates,
    )

    return await get_calibration_candidates(dive_id, session=session, **kw)


async def test_matching_same_camera_dive_with_extrinsics_is_returned(session):
    session.add_all([_dive(1, 5), _dive(2, 5, days=3)])
    await session.flush()
    session.add_all([_line(1, 0.0, -100.0), _line(2, 0.0, -100.0), _extrinsics(2, 5)])
    await session.flush()

    result = await _candidates(session, 1)
    assert [c.dive_id for c in result] == [2]
    assert result[0].line_angle_deg == pytest.approx(0.0, abs=1e-6)
    assert result[0].line_offset_px == pytest.approx(0.0, abs=1e-6)
    assert result[0].laser_extrinsics_id is not None
    assert result[0].days_apart == pytest.approx(3.0)


async def test_excludes_different_camera_no_extrinsics_and_out_of_tolerance(session):
    session.add_all(
        [_dive(1, 5), _dive(2, 9), _dive(3, 5), _dive(4, 5), _dive(5, 5)]
    )
    await session.flush()
    session.add_all(
        [
            _line(1, 0.0, -100.0),  # target
            _line(2, 0.0, -100.0),  # matching line but DIFFERENT camera (9)
            _extrinsics(2, 9),
            _line(3, 0.0, -100.0),  # matching line, same camera, NO extrinsics
            _line(4, 2.0, -100.0),  # same camera + extrinsics, angle 2deg > 1deg
            _extrinsics(4, 5),
            _line(5, 0.0, -200.0),  # same camera + extrinsics, offset 100px > 30
            _extrinsics(5, 5),
        ]
    )
    await session.flush()

    result = await _candidates(session, 1)
    assert not result


async def test_ranks_closer_line_first(session):
    session.add_all([_dive(1, 5), _dive(2, 5), _dive(3, 5)])
    await session.flush()
    session.add_all(
        [
            _line(1, 0.0, -100.0),  # target
            _line(2, 0.5, -110.0),  # 0.5deg, offset 10 -> ranked 2nd
            _extrinsics(2, 5),
            _line(3, 0.0, -100.0),  # exact match -> ranked 1st
            _extrinsics(3, 5),
        ]
    )
    await session.flush()

    # widen tolerance so the 0.5deg candidate is admitted (it's outside the
    # tightened default) — this test is about ordering, not the gate.
    result = await _candidates(session, 1, max_angle_deg=1.0, max_offset_px=30.0)
    assert [c.dive_id for c in result] == [3, 2]


async def test_default_tolerance_excludes_a_half_degree_match(session):
    # The tightened default (0.1deg) must reject a 0.5deg line match that the
    # old 1deg default would have (wrongly) admitted — measurement-grade gate.
    session.add_all([_dive(1, 5), _dive(2, 5)])
    await session.flush()
    session.add_all([_line(1, 0.0, -100.0), _line(2, 0.5, -100.0), _extrinsics(2, 5)])
    await session.flush()

    assert await _candidates(session, 1) == []  # excluded at default 0.1deg
    # ...but admitted when tolerance is explicitly widened.
    assert [c.dive_id for c in await _candidates(session, 1, max_angle_deg=1.0)] == [2]


async def test_low_confidence_candidate_excluded(session):
    session.add_all([_dive(1, 5), _dive(2, 5)])
    await session.flush()
    session.add_all(
        [_line(1, 0.0, -100.0), _line(2, 0.0, -100.0, confidence=1.0), _extrinsics(2, 5)]
    )
    await session.flush()

    assert not await _candidates(session, 1, min_confidence=5.0)


async def test_returns_empty_when_target_has_no_fingerprint(session):
    session.add_all([_dive(1, 5), _dive(2, 5)])
    await session.flush()
    session.add_all([_line(2, 0.0, -100.0), _extrinsics(2, 5)])  # target (1) has none
    await session.flush()

    assert not await _candidates(session, 1)


async def test_sign_flipped_line_still_matches(session):
    # (a,b,c) and (-a,-b,-c) are the same physical line; must compare equal.
    from fishsense_api.models.dive_laser_line import (
        DiveLaserLine,
    )

    session.add_all([_dive(1, 5), _dive(2, 5)])
    await session.flush()
    session.add(_line(1, 0.0, -100.0))  # normal (1,0), c=-100
    session.add(
        DiveLaserLine(  # same line, sign-flipped
            dive_id=2, a=-1.0, b=0.0, c=100.0,
            n_points=30, inlier_count=29, inlier_fraction=0.96,
            residual_std=0.5, label_noise_mad=0.4, line_confidence=5000.0,
        )
    )
    session.add(_extrinsics(2, 5))
    await session.flush()

    result = await _candidates(session, 1)
    assert [c.dive_id for c in result] == [2]
    assert result[0].line_angle_deg == pytest.approx(0.0, abs=1e-6)
    assert result[0].line_offset_px == pytest.approx(0.0, abs=1e-6)
