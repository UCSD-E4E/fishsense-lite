"""Cohort tests for the decoupled head/tail populate parent.

Which dives need head/tail LS tasks (re)populated. Prediction-gated, which is
what makes this the *decoupled* populate cohort rather than the preprocess one:
populate seeds sentinel `HeadTailLabel` rows and the predict cohort excludes
any image with a live label, so populating an unpredicted image would starve it
of a prediction forever.
"""

from __future__ import annotations

from tests_support.db import (  # noqa: F401
    dive as _dive,
    image as _image,
)


def _laser(label_id, image_id, completed=True, superseded=False):
    from fishsense_api.models.laser_label import LaserLabel

    return LaserLabel(
        id=label_id, image_id=image_id, x=1.0, y=2.0,
        completed=completed, superseded=superseded,
    )


def _prediction(pred_id, image_id, status="predicted"):
    from fishsense_api.models.head_tail_prediction import HeadTailPrediction

    return HeadTailPrediction(id=pred_id, image_id=image_id, status=status)


def _headtail(label_id, image_id, completed=True, superseded=False):
    from fishsense_api.models.head_tail_label import HeadTailLabel

    return HeadTailLabel(
        id=label_id, image_id=image_id, completed=completed,
        superseded=superseded, label_studio_project_id=7,
    )


async def _select(session):
    from fishsense_api.controllers.dive_prediction_cohort_controller import (
        select_dives_needing_headtail_population,
    )

    return await select_dives_needing_headtail_population(session=session)


async def test_selects_a_predicted_unlabelled_image(session):
    session.add_all([_dive(1), _image(11, 1), _laser(101, 11), _prediction(1, 11)])
    await session.flush()
    assert await _select(session) == [1]


async def test_unpredicted_images_are_not_populated(session):
    """The gate. Without it, populate would seed a sentinel row and remove the
    image from the predict cohort permanently."""
    session.add_all([_dive(1), _image(11, 1), _laser(101, 11)])
    await session.flush()
    assert await _select(session) == []


async def test_an_abstention_still_opens_the_gate(session):
    """The detector visited the image; withholding it forever would strand it
    with neither a prediction nor a human label."""
    session.add_all(
        [_dive(1), _image(11, 1), _laser(101, 11),
         _prediction(1, 11, status="no_detections")]
    )
    await session.flush()
    assert await _select(session) == [1]


async def test_drops_out_once_labelling_is_complete(session):
    session.add_all(
        [_dive(1), _image(11, 1), _laser(101, 11), _prediction(1, 11),
         _headtail(201, 11, completed=True)]
    )
    await session.flush()
    assert await _select(session) == []


async def test_stays_in_the_cohort_while_labelling_is_incomplete(session):
    """"No completed label", not "no row" — so the idempotent populate
    self-heals hourly until labelers finish."""
    session.add_all(
        [_dive(1), _image(11, 1), _laser(101, 11), _prediction(1, 11),
         _headtail(201, 11, completed=False)]
    )
    await session.flush()
    assert await _select(session) == [1]


async def test_requires_a_valid_laser(session):
    """Mirrors the predict cohort: no live dot, nothing this stage can do."""
    session.add_all(
        [_dive(1), _image(11, 1), _laser(101, 11, superseded=True), _prediction(1, 11)]
    )
    await session.flush()
    assert await _select(session) == []


async def test_only_high_priority_and_canonical(session):
    from fishsense_api.models.priority import Priority

    low = _dive(1)
    low.priority = Priority.LOW
    noncanon = _image(22, 2)
    noncanon.is_canonical = False
    session.add_all(
        [low, _dive(2), _image(11, 1), noncanon,
         _laser(101, 11), _laser(102, 22),
         _prediction(1, 11), _prediction(2, 22)]
    )
    await session.flush()
    assert await _select(session) == []


async def test_returns_every_match_in_id_order(session):
    session.add_all(
        [_dive(2), _dive(1), _image(22, 2), _image(11, 1),
         _laser(102, 22), _laser(101, 11),
         _prediction(2, 22), _prediction(1, 11)]
    )
    await session.flush()
    assert await _select(session) == [1, 2]
