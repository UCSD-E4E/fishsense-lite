"""Tests for `set_dive_slate` — the only write path for `Dive.dive_slate_id`.

Populated by the species-label sync from the labeler's slate-type choice
(and usable by an operator). FK-less in-memory sqlite, same as the
calibration-source endpoint tests — we exercise the controller directly.
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


def _dive(dive_id: int, *, dive_slate_id=None):
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    return Dive(
        id=dive_id,
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority=Priority.HIGH,
        dive_slate_id=dive_slate_id,
    )


def _dive_slate(slate_id: int):
    from fishsense_api.models.dive_slate import DiveSlate  # pylint: disable=import-outside-toplevel

    return DiveSlate(
        id=slate_id,
        name=f"V-Slate {slate_id}",
        dpi=300,
        path=f"/dev/null/slate-{slate_id}.pdf",
    )


async def test_set_dive_slate_sets_the_fk(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        set_dive_slate,
    )
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    session.add(_dive_slate(9))
    await session.flush()

    returned = await set_dive_slate(1, 9, session=session)
    assert returned == 1
    assert (await session.get(Dive, 1)).dive_slate_id == 9


async def test_set_dive_slate_overwrites_existing(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        set_dive_slate,
    )
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel

    session.add(_dive(1, dive_slate_id=8))
    session.add_all([_dive_slate(8), _dive_slate(9)])
    await session.flush()

    await set_dive_slate(1, 9, session=session)
    assert (await session.get(Dive, 1)).dive_slate_id == 9


async def test_set_dive_slate_404_when_dive_missing(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        set_dive_slate,
    )

    session.add(_dive_slate(9))
    await session.flush()

    with pytest.raises(HTTPException) as exc:
        await set_dive_slate(999, 9, session=session)
    assert exc.value.status_code == 404


async def test_set_dive_slate_404_when_template_missing(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        set_dive_slate,
    )

    session.add(_dive(1))
    await session.flush()

    with pytest.raises(HTTPException) as exc:
        await set_dive_slate(1, 999, session=session)
    assert exc.value.status_code == 404
