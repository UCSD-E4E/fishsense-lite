"""Cohort tests for the head/tail predict stage's dive selector.

The stage predicts snout/fork for images that already carry a validated laser
dot and have not been head/tail labelled yet. The interesting cases are all
about *when a dive leaves the cohort* — a selector that never goes false
re-stages the same dive every hour forever, which is the failure mode
CLAUDE.md records for stage 14 before the `measured` rescope.
"""

from __future__ import annotations

from tests_support.db import (  # noqa: F401
    dive as _dive,
    image as _image,
)


def _laser(label_id, image_id, *, completed=True, superseded=False, x=100.0, y=200.0):
    from fishsense_api.models.laser_label import LaserLabel

    return LaserLabel(
        id=label_id,
        image_id=image_id,
        x=x,
        y=y,
        completed=completed,
        superseded=superseded,
    )


def _headtail(label_id, image_id, *, completed=True, superseded=False, project_id=7):
    from fishsense_api.models.head_tail_label import HeadTailLabel

    return HeadTailLabel(
        id=label_id,
        image_id=image_id,
        completed=completed,
        superseded=superseded,
        label_studio_project_id=project_id,
        head_x=1.0,
        head_y=2.0,
        tail_x=3.0,
        tail_y=4.0,
    )


def _prediction(pred_id, image_id, *, version=None, laser_label_id=None):
    from fishsense_api.models.head_tail_prediction import HeadTailPrediction
    from fishsense_shared.headtail_predictor import HEADTAIL_PREDICTOR_VERSION

    return HeadTailPrediction(
        id=pred_id,
        image_id=image_id,
        predictor_version=(
            HEADTAIL_PREDICTOR_VERSION if version is None else version
        ),
        laser_label_id=laser_label_id,
        status="predicted",
    )


async def _select(session):
    from fishsense_api.controllers.dive_prediction_cohort_controller import (
        select_next_for_headtail_prediction,
    )

    return await select_next_for_headtail_prediction(session=session)


async def test_selects_a_dive_with_a_valid_laser_and_no_prediction(session):
    session.add_all([_dive(1), _image(11, 1), _laser(101, 11)])
    await session.flush()
    assert await _select(session) == 1


async def test_ignores_a_dive_with_no_valid_laser(session):
    """The whole stage is gated on the dot: no laser, nothing to crop around."""
    session.add_all([_dive(1), _image(11, 1), _laser(101, 11, completed=False)])
    await session.flush()
    assert await _select(session) is None


async def test_ignores_a_superseded_laser(session):
    """A label RANSAC dead-lettered is not a live dot."""
    session.add_all([_dive(1), _image(11, 1), _laser(101, 11, superseded=True)])
    await session.flush()
    assert await _select(session) is None


async def test_drops_out_once_predicted(session):
    """The cohort must go false, or the dive is re-selected every hour."""
    session.add_all(
        [_dive(1), _image(11, 1), _laser(101, 11), _prediction(1, 11, laser_label_id=101)]
    )
    await session.flush()
    assert await _select(session) is None


async def test_never_predicts_over_a_completed_human_label(session):
    session.add_all([_dive(1), _image(11, 1), _laser(101, 11), _headtail(201, 11)])
    await session.flush()
    assert await _select(session) is None


async def test_a_superseded_human_label_re_enters_the_cohort(session):
    """A dead-lettered head/tail label leaves the image with no live human
    work, so it should be predicted again."""
    session.add_all(
        [_dive(1), _image(11, 1), _laser(101, 11), _headtail(201, 11, superseded=True)]
    )
    await session.flush()
    assert await _select(session) == 1


async def test_stale_predictor_version_re_enters_the_cohort(session):
    """Mismatch, not absence — the point of `predictor_version`.

    `IS DISTINCT FROM`, not `!=`: a pre-versioning row is NULL, and `!=` would
    answer NULL and select nothing.
    """
    session.add_all(
        [_dive(1), _image(11, 1), _laser(101, 11), _prediction(1, 11, version=0)]
    )
    await session.flush()
    assert await _select(session) == 1


async def test_null_predictor_version_re_enters_the_cohort(session):
    session.add_all(
        [_dive(1), _image(11, 1), _laser(101, 11), _prediction(1, 11, version=0)]
    )
    await session.flush()
    assert await _select(session) == 1


async def test_prediction_from_a_superseded_laser_is_stale(session):
    """The provenance clause: the dot that chose the fish was later
    dead-lettered, so the mask may be of the wrong thing entirely.

    Selecting on this is what makes a supersede pass drain instead of leaving
    stale predictions sitting unnoticed — the LaserDepth lesson.
    """
    session.add_all(
        [
            _dive(1),
            _image(11, 1),
            _laser(101, 11, superseded=True),
            _laser(102, 11),  # a live dot remains, so the image is still eligible
            _prediction(1, 11, laser_label_id=101),
        ]
    )
    await session.flush()
    assert await _select(session) == 1


async def test_only_high_priority(session):
    from fishsense_api.models.priority import Priority

    dive = _dive(1)
    dive.priority = Priority.LOW
    session.add_all([dive, _image(11, 1), _laser(101, 11)])
    await session.flush()
    assert await _select(session) is None


async def test_non_canonical_images_are_ignored(session):
    image = _image(11, 1)
    image.is_canonical = False
    session.add_all([_dive(1), image, _laser(101, 11)])
    await session.flush()
    assert await _select(session) is None


async def test_returns_the_lowest_dive_id_first(session):
    session.add_all(
        [
            _dive(2),
            _dive(1),
            _image(22, 2),
            _image(11, 1),
            _laser(102, 22),
            _laser(101, 11),
        ]
    )
    await session.flush()
    assert await _select(session) == 1


async def test_first_prediction_is_preferred_over_an_upgrade(session):
    """A dive needing a *first* prediction outranks one needing only an
    upgrade, even with a higher id.

    This is what stops the upgrade queue starving new work. A fallback-tier
    row (`HEADTAIL_FALLBACK_PREDICTOR_VERSION`) is permanently stale by
    design, so while no GPU is available such a dive never leaves the cohort.
    Ordered by id alone it would be selected every firing, the GPU-less worker
    would skip every image as unimprovable, and every dive behind it with
    genuinely unpredicted images would wait -- the dive-60 stall shape,
    reached from a different direction.
    """
    from fishsense_shared.headtail_predictor import (
        HEADTAIL_FALLBACK_PREDICTOR_VERSION,
    )

    session.add_all(
        [
            _dive(1),
            _image(11, 1),
            _laser(101, 11),
            # dive 1 is fully fallback-predicted: stale, but not improvable
            # without a GPU.
            _prediction(1001, 11, version=HEADTAIL_FALLBACK_PREDICTOR_VERSION),
            _dive(2),
            _image(22, 2),
            _laser(102, 22),  # never predicted
        ]
    )
    await session.flush()

    assert await _select(session) == 2


async def test_an_upgrade_only_dive_is_still_selected_when_nothing_else_needs_one(
    session,
):
    """Preference, not exclusion. Once every first prediction is done the
    upgrade queue must still drain, or a GPU coming back would change
    nothing."""
    from fishsense_shared.headtail_predictor import (
        HEADTAIL_FALLBACK_PREDICTOR_VERSION,
    )

    session.add_all(
        [
            _dive(1),
            _image(11, 1),
            _laser(101, 11),
            _prediction(1001, 11, version=HEADTAIL_FALLBACK_PREDICTOR_VERSION),
        ]
    )
    await session.flush()

    assert await _select(session) == 1


async def test_a_fallback_row_is_permanently_stale(session):
    """The upgrade queue itself: a fallback-tier row must never read as
    current, whatever `HEADTAIL_PREDICTOR_VERSION` is bumped to."""
    from fishsense_shared.headtail_predictor import (
        HEADTAIL_FALLBACK_PREDICTOR_VERSION,
        HEADTAIL_PREDICTOR_VERSION,
    )

    assert HEADTAIL_FALLBACK_PREDICTOR_VERSION != HEADTAIL_PREDICTOR_VERSION
    assert HEADTAIL_FALLBACK_PREDICTOR_VERSION < 0, (
        "negative so it cannot collide with any future forward bump"
    )

    session.add_all(
        [
            _dive(1),
            _image(11, 1),
            _laser(101, 11),
            _prediction(1001, 11, version=HEADTAIL_FALLBACK_PREDICTOR_VERSION),
        ]
    )
    await session.flush()

    assert await _select(session) == 1, "fallback rows must re-enter the cohort"
