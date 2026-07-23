"""Tests for the dive calibration-source link.

`Dive.calibration_dive_id` lets a fish-only dive (no slate frames of its
own) borrow a sibling slate dive's laser calibration. These pin down:

  * `get_laser_extrinsics_for_dive` resolves own calibration first, then
    falls back through the link, then 404s.
  * `set_dive_calibration_source` / `clear_dive_calibration_source`
    manage the link (with self-link + missing-dive guards).

FK-less in-memory sqlite, same as test_select_next_dive_endpoints.py — we
exercise the controller functions directly, not referential integrity.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
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


def _dive(dive_id: int, *, calibration_dive_id=None):
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    return Dive(
        id=dive_id,
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority=Priority.HIGH,
        calibration_dive_id=calibration_dive_id,
    )


def _extrinsics(dive_id: int, *, position, created_at):
    from fishsense_api.models.laser_extrinsics import LaserExtrinsics  # pylint: disable=import-outside-toplevel

    return LaserExtrinsics(
        dive_id=dive_id,
        camera_id=1,
        laser_position=position,
        laser_axis=[0.0, 0.0, 1.0],
        created_at=created_at,
    )


# ---------- resolution ----------


async def test_own_extrinsics_win_over_the_link(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        get_laser_extrinsics_for_dive,
    )

    session.add_all([_dive(1), _dive(2, calibration_dive_id=1)])
    await session.flush()
    session.add(
        _extrinsics(1, position=[9.0, 9.0, 9.0],
                    created_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    )
    session.add(
        _extrinsics(2, position=[2.0, 2.0, 2.0],
                    created_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    )
    await session.flush()

    result = await get_laser_extrinsics_for_dive(2, session=session)
    assert result.laser_position == [2.0, 2.0, 2.0]  # dive 2's own, not dive 1's


async def test_falls_back_to_linked_source_when_no_own_extrinsics(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        get_laser_extrinsics_for_dive,
    )

    session.add_all([_dive(1), _dive(2, calibration_dive_id=1)])
    await session.flush()
    session.add(
        _extrinsics(1, position=[1.0, 1.0, 1.0],
                    created_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    )
    await session.flush()

    result = await get_laser_extrinsics_for_dive(2, session=session)
    assert result.laser_position == [1.0, 1.0, 1.0]  # borrowed from dive 1


async def test_404_when_neither_own_nor_linked_extrinsics(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        get_laser_extrinsics_for_dive,
    )

    session.add_all([_dive(1), _dive(2, calibration_dive_id=1)])
    await session.flush()

    with pytest.raises(HTTPException) as exc:
        await get_laser_extrinsics_for_dive(2, session=session)
    assert exc.value.status_code == 404


async def test_no_link_and_no_own_extrinsics_is_404(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        get_laser_extrinsics_for_dive,
    )

    session.add(_dive(1))
    await session.flush()

    with pytest.raises(HTTPException) as exc:
        await get_laser_extrinsics_for_dive(1, session=session)
    assert exc.value.status_code == 404


# ---------- set / clear ----------


async def test_set_calibration_source_links_the_dives(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        set_dive_calibration_source,
    )
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1), _dive(2)])
    await session.flush()

    returned = await set_dive_calibration_source(2, 1, session=session)
    assert returned == 2
    assert (await session.get(Dive, 2)).calibration_dive_id == 1


async def test_set_calibration_source_rejects_self_link(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        set_dive_calibration_source,
    )

    session.add(_dive(1))
    await session.flush()

    with pytest.raises(HTTPException) as exc:
        await set_dive_calibration_source(1, 1, session=session)
    assert exc.value.status_code == 400


async def test_set_calibration_source_404_when_source_missing(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        set_dive_calibration_source,
    )

    session.add(_dive(1))
    await session.flush()

    with pytest.raises(HTTPException) as exc:
        await set_dive_calibration_source(1, 999, session=session)
    assert exc.value.status_code == 404


async def test_set_calibration_source_404_when_dive_missing(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        set_dive_calibration_source,
    )

    session.add(_dive(1))
    await session.flush()

    with pytest.raises(HTTPException) as exc:
        await set_dive_calibration_source(999, 1, session=session)
    assert exc.value.status_code == 404


async def test_clear_calibration_source_unlinks(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        clear_dive_calibration_source,
    )
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1), _dive(2, calibration_dive_id=1)])
    await session.flush()

    await clear_dive_calibration_source(2, session=session)
    assert (await session.get(Dive, 2)).calibration_dive_id is None


async def test_clear_calibration_source_is_idempotent(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        clear_dive_calibration_source,
    )
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()

    await clear_dive_calibration_source(1, session=session)  # already null
    assert (await session.get(Dive, 1)).calibration_dive_id is None


async def test_clear_calibration_source_404_when_dive_missing(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        clear_dive_calibration_source,
    )

    with pytest.raises(HTTPException) as exc:
        await clear_dive_calibration_source(999, session=session)
    assert exc.value.status_code == 404
