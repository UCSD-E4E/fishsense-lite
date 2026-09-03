"""Tests for the head/tail-prediction endpoints (model-assisted labeling store).

FK-less in-memory sqlite, same as the other controller tests — the controller
functions are exercised directly.

The store mirrors `LaserPrediction`: its own table, one row per image, upserted
on the natural key, and read by the head/tail populate step to seed Label
Studio pre-annotations. Kept separate from `HeadTailLabel` so a prediction can
never count toward a human "valid head/tail" gate.
"""

from __future__ import annotations

from sqlmodel import select

# Shared with the other controller tests — see `tests_support.db`.
from tests_support.db import (  # noqa: F401
    dive as _dive,
    image as _image,
)


def _prediction(*, head=(10.0, 20.0), tail=(110.0, 25.0), **kwargs):
    from fishsense_api.models.head_tail_prediction import HeadTailPrediction

    return HeadTailPrediction(
        head_x=head[0],
        head_y=head[1],
        tail_x=tail[0],
        tail_y=tail[1],
        width=4014,
        height=3016,
        **kwargs,
    )


async def test_put_creates_a_prediction(session):
    from fishsense_api.controllers.head_tail_prediction_controller import (
        put_head_tail_prediction,
    )
    from fishsense_api.models.head_tail_prediction import HeadTailPrediction

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    pid = await put_head_tail_prediction(11, _prediction(), session=session)
    row = await session.get(HeadTailPrediction, pid)
    assert row.image_id == 11
    assert (row.head_x, row.head_y, row.tail_x, row.tail_y) == (10.0, 20.0, 110.0, 25.0)
    assert (row.width, row.height) == (4014, 3016)
    assert row.status == "predicted"


async def test_put_upserts_on_image_id(session):
    """A re-run of the predict stage must overwrite, not duplicate.

    `merge` on `id=None` always INSERTs, which would violate
    `uq_headtail_prediction_image` — the natural key has to be resolved first.
    This is the bug all four `put_*_label` handlers shipped with.
    """
    from fishsense_api.controllers.head_tail_prediction_controller import (
        put_head_tail_prediction,
    )
    from fishsense_api.models.head_tail_prediction import HeadTailPrediction

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    first = await put_head_tail_prediction(11, _prediction(), session=session)
    second = await put_head_tail_prediction(
        11, _prediction(head=(1.0, 1.0), tail=(2.0, 2.0)), session=session
    )

    assert first == second
    rows = (
        await session.exec(
            select(HeadTailPrediction).where(HeadTailPrediction.image_id == 11)
        )
    ).all()
    assert len(rows) == 1
    assert (rows[0].head_x, rows[0].tail_x) == (1.0, 2.0)


async def test_get_by_image_returns_none_when_absent(session):
    from fishsense_api.controllers.head_tail_prediction_controller import (
        get_head_tail_prediction,
    )

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    assert await get_head_tail_prediction(11, session=session) is None


async def test_get_for_dive_returns_only_that_dives_rows(session):
    from fishsense_api.controllers.head_tail_prediction_controller import (
        get_head_tail_predictions_for_dive,
        put_head_tail_prediction,
    )

    session.add_all([_dive(1), _dive(2), _image(11, 1), _image(22, 2)])
    await session.flush()
    await put_head_tail_prediction(11, _prediction(), session=session)
    await put_head_tail_prediction(22, _prediction(), session=session)

    rows = await get_head_tail_predictions_for_dive(1, session=session)
    assert [r.image_id for r in rows] == [11]


async def test_abstention_is_recorded_with_a_reason(session):
    """An abstention is a row, not a missing row.

    "The model found nothing" and "the laser landed on no fish" are different
    facts, and the cohort selects on the row's absence — so an abstention that
    wrote nothing would be re-predicted forever. Same reason `LaserPrediction`
    carries `rejected_out_of_region`.
    """
    from fishsense_api.controllers.head_tail_prediction_controller import (
        put_head_tail_prediction,
    )
    from fishsense_api.models.head_tail_prediction import HeadTailPrediction

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    pid = await put_head_tail_prediction(
        11,
        HeadTailPrediction(status="laser_off_all_fish"),
        session=session,
    )
    row = await session.get(HeadTailPrediction, pid)
    assert row.status == "laser_off_all_fish"
    assert row.head_x is None and row.tail_x is None


async def test_crop_origin_round_trips(session):
    """`crop_x/crop_y` are provenance for the laser-centred window (plan §4.3).

    The keypoints are stored already lifted into rectified-frame pixels; the
    origin is kept so a suspect prediction can be re-examined in the exact
    window the model saw, and so a change of crop size is visible in the data.
    """
    from fishsense_api.controllers.head_tail_prediction_controller import (
        put_head_tail_prediction,
    )
    from fishsense_api.models.head_tail_prediction import HeadTailPrediction

    session.add_all([_dive(1), _image(11, 1)])
    await session.flush()

    pid = await put_head_tail_prediction(
        11, _prediction(crop_x=1200, crop_y=900), session=session
    )
    row = await session.get(HeadTailPrediction, pid)
    assert (row.crop_x, row.crop_y) == (1200, 900)
