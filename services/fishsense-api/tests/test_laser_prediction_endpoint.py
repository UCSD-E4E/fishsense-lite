"""Tests for the laser-prediction endpoints (model-assisted labeling store).

FK-less in-memory sqlite, same as the other controller tests — we exercise
the controller functions directly.
"""

from __future__ import annotations


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


# --- auto-accept verdict ------------------------------------------------------
#
# The gate runs on the data-worker (numpy + the RANSAC kernel live there) and
# the laser populate step on the api-worker consumes the answer, so the verdict
# has to survive a round trip through the store. These fields are what carries
# it. `line_offset_px` / `line_position_z` are recorded but never re-decided on
# — the same role `LaserDepth.residual_m` plays — so an auto-accepted frame can
# be re-examined by margin later, not just by outcome.


async def test_put_round_trips_the_auto_accept_verdict(session):
    from fishsense_api.controllers.laser_prediction_controller import (
        put_laser_prediction,
    )
    from fishsense_api.models.laser_prediction import LaserPrediction

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    prediction = _prediction()
    prediction.auto_accept = True
    prediction.gate_verdict = "auto_accepted"
    prediction.line_offset_px = 1.25
    prediction.line_position_z = 0.8
    pid = await put_laser_prediction(11, prediction, session=session)

    row = await session.get(LaserPrediction, pid)
    assert row.auto_accept is True
    assert row.gate_verdict == "auto_accepted"
    assert (row.line_offset_px, row.line_position_z) == (1.25, 0.8)


async def test_auto_accept_defaults_to_false_for_an_ungated_prediction(session):
    """A prediction the gate has not seen must never read as auto-acceptable.
    The populate step keys on this, so the default is the safe direction."""
    from fishsense_api.controllers.laser_prediction_controller import (
        put_laser_prediction,
    )
    from fishsense_api.models.laser_prediction import LaserPrediction

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    pid = await put_laser_prediction(11, _prediction(), session=session)
    row = await session.get(LaserPrediction, pid)
    assert row.auto_accept is False
    assert row.gate_verdict is None


async def test_re_prediction_clears_a_stale_verdict(session):
    """Re-predicting an image invalidates the verdict that was computed from
    the old dot — the gate has to run again over the new prediction set. The
    upsert must not leave the previous `auto_accept=True` standing beside a
    coordinate it was never computed from."""
    from fishsense_api.controllers.laser_prediction_controller import (
        put_laser_prediction,
    )
    from fishsense_api.models.laser_prediction import LaserPrediction

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    gated = _prediction(x=100.0, y=200.0)
    gated.auto_accept = True
    gated.gate_verdict = "auto_accepted"
    await put_laser_prediction(11, gated, session=session)

    pid = await put_laser_prediction(11, _prediction(x=700.0, y=900.0), session=session)
    row = await session.get(LaserPrediction, pid)
    assert (row.x, row.y) == (700.0, 900.0)
    assert row.auto_accept is False
    assert row.gate_verdict is None
