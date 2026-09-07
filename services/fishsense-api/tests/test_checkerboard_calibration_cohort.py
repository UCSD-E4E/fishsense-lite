"""Cohort selector for checkerboard laser calibration.

The sibling of stage 13's `select_next_for_laser_calibration`, and the shape
of the difference is the point: a slate observation is a hand-clicked
`DiveSlateLabel` whose image also carries a live dot, while a checkerboard
observation is just **a canonical image with a live laser dot**. The board's
corners are detected by the data-worker at run time, so there is no label row
to count and nothing for SQL to check about the board itself.

That makes this selector a deliberate over-approximation. It is documented in
the selector and pinned by `test_a_dive_whose_frames_hide_the_board_is_still_offered`
below, so nobody later reads a green cohort as a promise the board was found.
"""

from __future__ import annotations

import pytest

from tests_support.db import (  # noqa: F401
    dive as _dive,
    image as _image,
)


def _target(target_id: int = 1, name: str = "E4E Checkerboard"):
    from fishsense_api.models.calibration_target import CalibrationTarget

    return CalibrationTarget(
        id=target_id, name=name, rows=10, cols=14, square_size_m=0.0254
    )


def _dot(image_id: int, *, x: float | None = 100.0, superseded: bool = False):
    """A live laser dot — one checkerboard calibration observation.

    `superseded == False` and nothing about `completed`, because that is
    exactly what `get_laser_label` filters on: a populate-seeded placeholder
    with NULL x/y is excluded by the x/y check, not by its completion state.
    """
    from fishsense_api.models.laser_label import LaserLabel

    return LaserLabel(image_id=image_id, x=x, y=200.0, superseded=superseded)


def _extrinsics(dive_id: int):
    from fishsense_api.models.laser_extrinsics import LaserExtrinsics

    return LaserExtrinsics(
        dive_id=dive_id,
        camera_id=1,
        laser_position=[0.0, 0.1, 0.0],
        laser_axis=[0.0, 0.0, 1.0],
    )


async def _seed(session, dives, targets=(1,)):
    """`dives` is `(dive_id, calibration_target_id, dot_count)` triples."""
    for target_id in targets:
        session.add(_target(target_id))
    session.add_all(
        [_dive(d, calibration_target_id=t) for d, t, _ in dives],
    )
    await session.flush()
    for dive_id, _, dots in dives:
        for n in range(dots):
            image_id = dive_id * 100 + n
            session.add(_image(image_id, dive_id))
            await session.flush()
            session.add(_dot(image_id))
    await session.flush()


async def test_picks_a_linked_dive_with_enough_live_dots(session):
    from fishsense_api.controllers.dive_cohort_controller import (
        select_next_for_checkerboard_laser_calibration,
    )

    await _seed(session, [(1, None, 5), (2, 1, 5)])

    assert await select_next_for_checkerboard_laser_calibration(session=session) == 2


async def test_requires_min_observations(session):
    """Below `MIN_SLATE_LASER_POINTS` the fit is refused, so do not offer it.

    One threshold spelled on both sides of the worker boundary. A dive that
    clears the cohort's copy but not the activity's is re-selected hourly with
    nothing written — the dive 347 wedge, which also blocked every higher-id
    dive behind it.
    """
    from fishsense_api.controllers.dive_cohort_controller import (
        select_next_for_checkerboard_laser_calibration,
    )

    await _seed(session, [(1, 1, 1), (2, 1, 2)])

    assert await select_next_for_checkerboard_laser_calibration(session=session) == 2


async def test_skips_a_dive_that_already_has_its_own_extrinsics(session):
    from fishsense_api.controllers.dive_cohort_controller import (
        select_next_for_checkerboard_laser_calibration,
    )

    await _seed(session, [(1, 1, 5), (2, 1, 5)])
    session.add(_extrinsics(1))
    await session.flush()

    assert await select_next_for_checkerboard_laser_calibration(session=session) == 2


async def test_a_borrowed_calibration_does_not_exclude_a_dive(session):
    """`calibration_dive_id` is not a substitute for calibrating yourself.

    Stage 14's cohort accepts a borrowed calibration because it only needs
    *some* extrinsics to measure with. This one must not: a linked slate dive
    that can now fit its own is exactly the dive to fit, and the resolution
    order is own-wins-then-link, so its own answer takes over the moment it
    exists.
    """
    from fishsense_api.controllers.dive_cohort_controller import (
        select_next_for_checkerboard_laser_calibration,
    )
    from fishsense_api.models.dive import Dive

    await _seed(session, [(1, 1, 5)])
    session.add(_extrinsics(9))
    dive = await session.get(Dive, 1)
    dive.calibration_dive_id = 9
    session.add(dive)
    await session.flush()

    assert await select_next_for_checkerboard_laser_calibration(session=session) == 1


async def test_ignores_dives_with_no_calibration_target(session):
    """No link means no known scale, so there is nothing to fit against."""
    from fishsense_api.controllers.dive_cohort_controller import (
        select_next_for_checkerboard_laser_calibration,
    )

    await _seed(session, [(1, None, 9)])

    assert await select_next_for_checkerboard_laser_calibration(session=session) is None


async def test_ignores_non_high_priority_dives(session):
    from fishsense_api.controllers.dive_cohort_controller import (
        select_next_for_checkerboard_laser_calibration,
    )
    from fishsense_api.models.priority import Priority

    session.add(_target(1))
    session.add(_dive(1, priority=Priority.NONE, calibration_target_id=1))
    await session.flush()
    for image_id in (101, 102, 103):
        session.add(_image(image_id, 1))
        await session.flush()
        session.add(_dot(image_id))
    await session.flush()

    assert await select_next_for_checkerboard_laser_calibration(session=session) is None


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"x": None}, "populate-seeded placeholder, no dot"),
        ({"superseded": True}, "dead-lettered by the RANSAC validator"),
    ],
)
async def test_unusable_laser_labels_do_not_count(session, kwargs, why):
    from fishsense_api.controllers.dive_cohort_controller import (
        select_next_for_checkerboard_laser_calibration,
    )

    session.add(_target(1))
    session.add(_dive(1, calibration_target_id=1))
    await session.flush()
    session.add_all([_image(101, 1), _image(102, 1), _image(103, 1)])
    await session.flush()
    # Two live dots would be enough; make two of the three unusable.
    session.add_all([_dot(101), _dot(102, **kwargs), _dot(103, **kwargs)])
    await session.flush()

    assert (
        await select_next_for_checkerboard_laser_calibration(session=session) is None
    ), why


async def test_non_canonical_images_do_not_count(session):
    """Duplicate rows are invisible to every other cohort; same here.

    Two of the 2023.08.18 dives are wholly or partly duplicate content, so
    this is not hypothetical for exactly the corpus this stage serves.
    """
    from fishsense_api.controllers.dive_cohort_controller import (
        select_next_for_checkerboard_laser_calibration,
    )

    session.add(_target(1))
    session.add(_dive(1, calibration_target_id=1))
    await session.flush()
    session.add_all(
        [_image(101, 1), _image(102, 1, is_canonical=False)]
    )
    await session.flush()
    session.add_all([_dot(101), _dot(102)])
    await session.flush()

    assert await select_next_for_checkerboard_laser_calibration(session=session) is None


async def test_a_dive_whose_frames_hide_the_board_is_still_offered(session):
    """The known over-approximation, pinned so it stays known.

    SQL cannot tell whether `findChessboardCornersSB` will find the board:
    that needs the raw bytes, a rectification and a detector run. So a dive
    linked to a target whose frames never detect one is offered here, the
    activity refuses it, and it is re-selected every hour — blocking every
    higher-id dive behind it, because this selector is ORDER BY id LIMIT 1.

    The remedy is operator-side and deliberate: clear the link
    (`DELETE /dives/{id}/calibration-target/`) or park the dive at
    `Priority.NONE` with a note saying why. Both drop it immediately.
    """
    from fishsense_api.controllers.dive_cohort_controller import (
        select_next_for_checkerboard_laser_calibration,
    )

    # Nothing here says anything about a checkerboard being visible — and
    # that is the whole content of this test.
    await _seed(session, [(1, 1, 4)])

    assert await select_next_for_checkerboard_laser_calibration(session=session) == 1


async def test_drains_in_dive_id_order(session):
    from fishsense_api.controllers.dive_cohort_controller import (
        select_next_for_checkerboard_laser_calibration,
    )

    await _seed(session, [(5, 1, 3), (2, 1, 3), (9, 1, 3)])

    assert await select_next_for_checkerboard_laser_calibration(session=session) == 2
