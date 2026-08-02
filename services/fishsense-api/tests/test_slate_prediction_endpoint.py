"""Tests for the slate-prediction endpoints (model-assisted labeling store).

FK-less in-memory sqlite, same as the other controller tests — we exercise the
controller functions directly.
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
    from fishsense_api.models.priority import (  # pylint: disable=import-outside-toplevel
        Priority,
    )

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
        checksum=f"img-{image_id:032d}"[:32],
        dive_id=dive_id,
    )


def _prediction(*, reference_points=None, confidence=0.9, rejected_reason=None):
    from fishsense_api.models.slate_prediction import (  # pylint: disable=import-outside-toplevel
        SlatePrediction,
    )

    return SlatePrediction(
        reference_points=reference_points,
        confidence=confidence,
        rejected_reason=rejected_reason,
        width=4014,
        height=3016,
    )


async def _count(session, image_id):
    from fishsense_api.models.slate_prediction import (  # pylint: disable=import-outside-toplevel
        SlatePrediction,
    )

    rows = (
        await session.exec(
            select(SlatePrediction).where(SlatePrediction.image_id == image_id)
        )
    ).all()
    return len(rows)


async def test_put_upserts_on_image_id(session):
    from fishsense_api.controllers.slate_prediction_controller import (  # pylint: disable=import-outside-toplevel
        get_slate_predictions_for_dive,
        put_slate_prediction,
    )

    session.add_all([_dive(1), _image(10, 1)])
    await session.flush()

    id1 = await put_slate_prediction(
        10, _prediction(reference_points=[[1.0, 2.0]], confidence=0.9), session=session
    )
    id2 = await put_slate_prediction(
        10, _prediction(reference_points=[[3.0, 4.0]], confidence=0.95), session=session
    )
    await session.flush()

    assert await _count(session, 10) == 1  # upsert, not append
    assert id1 == id2
    preds = await get_slate_predictions_for_dive(1, session=session)
    assert len(preds) == 1
    assert preds[0].reference_points == [[3.0, 4.0]]  # latest wins
    assert preds[0].confidence == pytest.approx(0.95)


async def test_put_stores_rejection_with_null_points(session):
    from fishsense_api.controllers.slate_prediction_controller import (  # pylint: disable=import-outside-toplevel
        get_slate_predictions_for_dive,
        put_slate_prediction,
    )

    session.add_all([_dive(1), _image(10, 1)])
    await session.flush()
    await put_slate_prediction(
        10,
        _prediction(reference_points=None, confidence=0.4, rejected_reason="low_confidence"),
        session=session,
    )
    await session.flush()

    preds = await get_slate_predictions_for_dive(1, session=session)
    assert preds[0].reference_points is None
    assert preds[0].rejected_reason == "low_confidence"


async def test_get_empty_when_dive_has_no_predictions(session):
    from fishsense_api.controllers.slate_prediction_controller import (  # pylint: disable=import-outside-toplevel
        get_slate_predictions_for_dive,
    )

    session.add(_dive(1))
    await session.flush()
    assert await get_slate_predictions_for_dive(1, session=session) == []
