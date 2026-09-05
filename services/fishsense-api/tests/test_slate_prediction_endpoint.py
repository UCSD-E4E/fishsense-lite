"""Tests for the slate-prediction endpoints (model-assisted labeling store).

FK-less in-memory sqlite, same as the other controller tests — we exercise the
controller functions directly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


import pytest
from sqlmodel import select

# Shared with the other controller tests — see `tests_support.db`.
from tests_support.db import (  # noqa: F401
    dive as _dive,
    image as _image,
)


def _prediction(*, reference_points=None, confidence=0.9, rejected_reason=None):
    from fishsense_api.models.slate_prediction import (
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
    from fishsense_api.models.slate_prediction import (
        SlatePrediction,
    )

    rows = (
        await session.exec(
            select(SlatePrediction).where(SlatePrediction.image_id == image_id)
        )
    ).all()
    return len(rows)


async def test_put_upserts_on_image_id(session):
    from fishsense_api.controllers.slate_prediction_controller import (
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
    from fishsense_api.controllers.slate_prediction_controller import (
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
    from fishsense_api.controllers.slate_prediction_controller import (
        get_slate_predictions_for_dive,
    )

    session.add(_dive(1))
    await session.flush()
    assert not await get_slate_predictions_for_dive(1, session=session)


async def test_put_stamps_created_at(session):
    """`created_at` is what makes a prediction datable, and nothing wrote it.

    Predictions upsert on `image_id`, so a re-prediction overwrites the
    coordinates a labeler was actually shown. The stamp is what separates the
    two cases: a prediction written *after* the label's `updated_at` provably
    never reached a labeler, and one written before it is the row that was
    shown. Without it, "did the labeler move the seeded point?" is unanswerable
    for the whole corpus.
    """
    from fishsense_api.controllers.slate_prediction_controller import (
        put_slate_prediction,
    )
    from fishsense_api.models.slate_prediction import SlatePrediction

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    before = datetime.now(timezone.utc)
    pid = await put_slate_prediction(11, _prediction(), session=session)
    row = await session.get(SlatePrediction, pid)

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
    from fishsense_api.controllers.slate_prediction_controller import (
        put_slate_prediction,
    )
    from fishsense_api.models.slate_prediction import SlatePrediction

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    ancient = datetime(2020, 1, 1, tzinfo=timezone.utc)
    stale = _prediction()
    stale.created_at = ancient
    pid = await put_slate_prediction(11, stale, session=session)
    row = await session.get(SlatePrediction, pid)

    stamped = row.created_at
    if stamped.tzinfo is None:
        stamped = stamped.replace(tzinfo=timezone.utc)
    assert stamped > ancient, "server-side stamp must overwrite a client value"
