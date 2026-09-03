"""Tests for the laser-prediction endpoints (model-assisted labeling store).

FK-less in-memory sqlite, same as the other controller tests — we exercise
the controller functions directly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


from sqlmodel import select

# Shared with the other controller tests — see `tests_support.db`.
from tests_support.db import (  # noqa: F401
    dive as _dive,
    image as _image,
)


def _prediction(*, x=1.0, y=2.0, confidence=0.9):
    from fishsense_api.models.laser_prediction import LaserPrediction

    return LaserPrediction(x=x, y=y, confidence=confidence)


async def test_put_creates_a_prediction(session):
    from fishsense_api.controllers.laser_prediction_controller import (
        put_laser_prediction,
    )
    from fishsense_api.models.laser_prediction import LaserPrediction

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    pid = await put_laser_prediction(11, _prediction(x=100.0, y=200.0), session=session)
    row = await session.get(LaserPrediction, pid)
    assert row.image_id == 11
    assert (row.x, row.y, row.confidence) == (100.0, 200.0, 0.9)


async def test_put_upserts_on_image_id(session):
    from fishsense_api.controllers.laser_prediction_controller import (
        put_laser_prediction,
    )
    from fishsense_api.models.laser_prediction import LaserPrediction

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
    from fishsense_api.controllers.laser_prediction_controller import (
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
    from fishsense_api.controllers.laser_prediction_controller import (
        get_laser_predictions_for_dive,
    )

    session.add(_dive(1))
    await session.flush()

    assert await get_laser_predictions_for_dive(1, session=session) == []


async def test_put_stamps_created_at(session):
    """`created_at` is what makes a prediction datable, and nothing wrote it.

    Predictions upsert on `image_id`, so a re-prediction overwrites the
    coordinates a labeler was actually shown. The stamp is what separates the
    two cases: a prediction written *after* the label's `updated_at` provably
    never reached a labeler, and one written before it is the row that was
    shown. Without it, "did the labeler move the seeded point?" is unanswerable
    for the whole corpus.
    """
    from fishsense_api.controllers.laser_prediction_controller import (
        put_laser_prediction,
    )
    from fishsense_api.models.laser_prediction import LaserPrediction

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    before = datetime.now(timezone.utc)
    pid = await put_laser_prediction(11, _prediction(), session=session)
    row = await session.get(LaserPrediction, pid)

    assert row.created_at is not None
    stamped = row.created_at
    if stamped.tzinfo is None:
        stamped = stamped.replace(tzinfo=timezone.utc)
    assert stamped >= before - timedelta(seconds=5)


async def test_put_restamps_and_ignores_a_client_supplied_created_at(session):
    """Stamped server-side, and re-stamped on upsert.

    The surviving row *is* the prediction in force, so the field means "when
    this x/y was produced". A client value must not win, or a replayed payload
    would date the new coordinates to the old run.
    """
    from fishsense_api.controllers.laser_prediction_controller import (
        put_laser_prediction,
    )
    from fishsense_api.models.laser_prediction import LaserPrediction

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    ancient = datetime(2020, 1, 1, tzinfo=timezone.utc)
    stale = _prediction()
    stale.created_at = ancient
    pid = await put_laser_prediction(11, stale, session=session)
    row = await session.get(LaserPrediction, pid)

    stamped = row.created_at
    if stamped.tzinfo is None:
        stamped = stamped.replace(tzinfo=timezone.utc)
    assert stamped > ancient, "server-side stamp must overwrite a client value"
