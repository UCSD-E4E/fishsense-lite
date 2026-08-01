"""Tests for `put_laser_extrinsics_for_dive` upsert + created_at semantics.

Regression for the #462-recovery bug: the calibration path inserted a new
`LaserExtrinsics` row on every call (merge with id=None) and never set
`created_at`. The latest-wins read orders by `created_at DESC`, and with
Postgres sorting NULLs first that resolved nothing -> 404. Two guarantees
pinned here:

  * put is an upsert keyed on `dive_id` (one row per dive, never a dupe),
  * put always stamps a non-NULL `created_at`,

so a recalibration overwrites in place and the read always resolves it.
FK-less in-memory sqlite, same harness as test_calibration_source_endpoints.py.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
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


def _extrinsics(dive_id: int, *, position, created_at=None):
    from fishsense_api.models.laser_extrinsics import (  # pylint: disable=import-outside-toplevel
        LaserExtrinsics,
    )

    return LaserExtrinsics(
        dive_id=dive_id,
        camera_id=1,
        laser_position=position,
        laser_axis=[0.0, 0.0, 1.0],
        created_at=created_at,
    )


async def _count_for_dive(session: AsyncSession, dive_id: int) -> int:
    from fishsense_api.models.laser_extrinsics import (  # pylint: disable=import-outside-toplevel
        LaserExtrinsics,
    )

    rows = (
        await session.exec(
            select(LaserExtrinsics).where(LaserExtrinsics.dive_id == dive_id)
        )
    ).all()
    return len(rows)


async def test_put_upserts_on_dive_id_no_duplicate(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        get_laser_extrinsics_for_dive,
        put_laser_extrinsics_for_dive,
    )

    first_id = await put_laser_extrinsics_for_dive(
        1, _extrinsics(1, position=[1.0, 1.0, 0.0]), session=session
    )
    second_id = await put_laser_extrinsics_for_dive(
        1, _extrinsics(1, position=[2.0, 2.0, 0.0]), session=session
    )
    await session.flush()

    assert await _count_for_dive(session, 1) == 1  # upsert, not append
    assert first_id == second_id  # same row reused
    result = await get_laser_extrinsics_for_dive(1, session=session)
    assert result.laser_position == [2.0, 2.0, 0.0]  # latest value wins


async def test_put_stamps_non_null_created_at(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        get_laser_extrinsics_for_dive,
        put_laser_extrinsics_for_dive,
    )

    # created_at intentionally omitted (None) by the caller, as the
    # calibration activity does.
    await put_laser_extrinsics_for_dive(
        1, _extrinsics(1, position=[1.0, 1.0, 0.0], created_at=None), session=session
    )
    await session.flush()

    result = await get_laser_extrinsics_for_dive(1, session=session)
    assert result.created_at is not None  # read resolves; never 404 on NULL


async def test_put_independent_across_dives(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        put_laser_extrinsics_for_dive,
    )

    await put_laser_extrinsics_for_dive(
        1, _extrinsics(1, position=[1.0, 1.0, 0.0]), session=session
    )
    await put_laser_extrinsics_for_dive(
        2, _extrinsics(2, position=[2.0, 2.0, 0.0]), session=session
    )
    await session.flush()

    assert await _count_for_dive(session, 1) == 1
    assert await _count_for_dive(session, 2) == 1
