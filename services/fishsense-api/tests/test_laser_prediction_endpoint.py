"""Tests for the laser-prediction endpoints (model-assisted labeling store).

FK-less in-memory sqlite, same as the other controller tests — we exercise
the controller functions directly.
"""

from __future__ import annotations

from datetime import datetime, timezone

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


def _dive(dive_id: int):
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    return Dive(
        id=dive_id,
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority=Priority.HIGH,
    )


def _image(image_id: int, dive_id: int):
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    return Image(
        id=image_id,
        path=f"/dev/null/img-{image_id}",
        taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        checksum=f"{image_id:032d}",
        dive_id=dive_id,
    )


def _prediction(*, x=1.0, y=2.0, confidence=0.9):
    from fishsense_api.models.laser_prediction import LaserPrediction  # pylint: disable=import-outside-toplevel

    return LaserPrediction(x=x, y=y, confidence=confidence)


async def test_put_creates_a_prediction(session):
    from fishsense_api.controllers.laser_prediction_controller import (  # pylint: disable=import-outside-toplevel
        put_laser_prediction,
    )
    from fishsense_api.models.laser_prediction import LaserPrediction  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    pid = await put_laser_prediction(11, _prediction(x=100.0, y=200.0), session=session)
    row = await session.get(LaserPrediction, pid)
    assert row.image_id == 11
    assert (row.x, row.y, row.confidence) == (100.0, 200.0, 0.9)


async def test_put_upserts_on_image_id(session):
    from fishsense_api.controllers.laser_prediction_controller import (  # pylint: disable=import-outside-toplevel
        put_laser_prediction,
    )
    from fishsense_api.models.laser_prediction import LaserPrediction  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    first = await put_laser_prediction(11, _prediction(x=1.0, y=1.0), session=session)
    second = await put_laser_prediction(
        11, _prediction(x=9.0, y=9.0, confidence=0.5), session=session
    )

    assert first == second  # same row reused, not a duplicate
    rows = (
        await session.exec(
            select(LaserPrediction).where(LaserPrediction.image_id == 11)
        )
    ).all()
    assert len(rows) == 1
    assert (rows[0].x, rows[0].y, rows[0].confidence) == (9.0, 9.0, 0.5)


async def test_get_returns_predictions_for_the_dive(session):
    from fishsense_api.controllers.laser_prediction_controller import (  # pylint: disable=import-outside-toplevel
        get_laser_predictions_for_dive,
        put_laser_prediction,
    )

    session.add_all([_dive(1), _dive(2), _image(11, 1), _image(12, 1), _image(21, 2)])
    await session.flush()
    await put_laser_prediction(11, _prediction(), session=session)
    await put_laser_prediction(12, _prediction(), session=session)
    await put_laser_prediction(21, _prediction(), session=session)  # other dive

    results = await get_laser_predictions_for_dive(1, session=session)
    assert {r.image_id for r in results} == {11, 12}


async def test_get_returns_empty_list_when_none(session):
    from fishsense_api.controllers.laser_prediction_controller import (  # pylint: disable=import-outside-toplevel
        get_laser_predictions_for_dive,
    )

    session.add(_dive(1))
    await session.flush()

    assert await get_laser_predictions_for_dive(1, session=session) == []
