"""Tests for the per-dive laser-line fingerprint endpoints.

`DiveLaserLine` persists the RANSAC/TLS laser line the validation already
fits, so `(camera_id, line)` becomes queryable (borrow candidates, drift,
mount-swap epochs, pooled calibration). Same upsert + non-NULL timestamp
guarantees as LaserExtrinsics, so a re-fit overwrites in place and the row
never carries a NULL `fitted_at`. FK-less in-memory sqlite harness.
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


def _line(dive_id: int, *, a, b, c, confidence=10.0):
    from fishsense_api.models.dive_laser_line import (  # pylint: disable=import-outside-toplevel
        DiveLaserLine,
    )

    return DiveLaserLine(
        dive_id=dive_id,
        a=a,
        b=b,
        c=c,
        n_points=20,
        inlier_count=19,
        inlier_fraction=0.95,
        residual_std=0.5,
        label_noise_mad=0.4,
        line_confidence=confidence,
    )


async def _count(session: AsyncSession, dive_id: int) -> int:
    from fishsense_api.models.dive_laser_line import (  # pylint: disable=import-outside-toplevel
        DiveLaserLine,
    )

    rows = (
        await session.exec(select(DiveLaserLine).where(DiveLaserLine.dive_id == dive_id))
    ).all()
    return len(rows)


async def test_put_upserts_on_dive_id(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        get_dive_laser_line,
        put_dive_laser_line,
    )

    id1 = await put_dive_laser_line(1, _line(1, a=1.0, b=0.0, c=-5.0), session=session)
    id2 = await put_dive_laser_line(1, _line(1, a=0.0, b=1.0, c=-9.0), session=session)
    await session.flush()

    assert await _count(session, 1) == 1  # upsert, not append
    assert id1 == id2
    got = await get_dive_laser_line(1, session=session)
    assert (got.a, got.b, got.c) == (0.0, 1.0, -9.0)  # latest wins


async def test_put_stamps_non_null_fitted_at(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        get_dive_laser_line,
        put_dive_laser_line,
    )

    await put_dive_laser_line(1, _line(1, a=1.0, b=0.0, c=-5.0), session=session)
    await session.flush()
    got = await get_dive_laser_line(1, session=session)
    assert got.fitted_at is not None


async def test_get_returns_none_when_absent(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        get_dive_laser_line,
    )

    assert await get_dive_laser_line(999, session=session) is None
