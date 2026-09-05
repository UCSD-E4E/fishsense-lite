"""Tests for the slate-prediction endpoints (model-assisted labeling store).

FK-less in-memory sqlite, same as the other controller tests — we exercise the
controller functions directly.
"""

from __future__ import annotations


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
    assert await get_slate_predictions_for_dive(1, session=session) == []
