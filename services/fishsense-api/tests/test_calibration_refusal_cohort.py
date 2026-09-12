"""A refused dive leaves the calibration cohort, and comes back when relabelled.

Both calibration cohorts select on dive *state* — "has no usable
`LaserExtrinsics` row" — and a refusal does not change that state. So without
this a dive whose observations cannot produce a sound fit is re-selected every
hour forever, re-staging its raw `.ORF`s from the NAS each time, and because
the selectors are `ORDER BY id LIMIT 1` it blocks every dive behind it. That is
the prod dive-347 shape, and the baseline gate made it reachable for eight more
dives at once.

The exclusion is scoped by *time* rather than by a flag, and that is the
property most worth pinning: a bare boolean would become a permanent exclusion
that outlives the problem, while a timestamp compared against the dive's labels
expires on its own the moment anybody relabels. Nobody has to remember to clear
it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests_support.db import (  # noqa: F401
    dive as _dive,
    image as _image,
)

BEFORE = datetime(2026, 9, 1, tzinfo=timezone.utc)
REFUSED = datetime(2026, 9, 10, tzinfo=timezone.utc)
AFTER = REFUSED + timedelta(hours=1)


def _laser_label(label_id: int, image_id: int, *, updated_at, x=600.0, y=500.0):
    from fishsense_api.models.laser_label import LaserLabel

    return LaserLabel(
        id=label_id,
        image_id=image_id,
        x=x,
        y=y,
        completed=True,
        superseded=False,
        updated_at=updated_at,
        label_studio_project_id=73,
    )


async def _seed(session, *, refused_at, label_updated_at, refused_labels_at=BEFORE):
    """A checkerboard dive with two dotted frames, optionally refused.

    `refused_labels_at` is the snapshot the refusal was computed from, not the
    wall clock it was recorded at — expiry compares that, so the two sides of
    the comparison both come from Label Studio. It defaults to `BEFORE`, the
    timestamp the labels carry at the moment of refusal.
    """
    from fishsense_api.models.calibration_target import CalibrationTarget

    session.add(
        CalibrationTarget(
            id=1, name="E4E Checkerboard", rows=10, cols=14, square_size_m=0.042
        )
    )
    dive = _dive(1, calibration_target_id=1)
    dive.calibration_refused_at = refused_at
    dive.calibration_refused_reason = "implausible baseline" if refused_at else None
    dive.calibration_refused_labels_at = refused_labels_at if refused_at else None
    session.add(dive)
    await session.flush()
    session.add_all([_image(10, 1), _image(11, 1)])
    await session.flush()
    session.add_all(
        [
            _laser_label(100, 10, updated_at=label_updated_at),
            _laser_label(101, 11, updated_at=label_updated_at),
        ]
    )
    await session.flush()


async def _select(session):
    from fishsense_api.controllers.dive_cohort_controller import (
        select_next_for_checkerboard_laser_calibration,
    )

    return await select_next_for_checkerboard_laser_calibration(session=session)


@pytest.mark.asyncio
async def test_a_dive_with_no_refusal_is_offered(session):
    """The control: without a refusal the cohort behaves as it always did."""
    await _seed(session, refused_at=None, label_updated_at=BEFORE)

    assert await _select(session) == 1


@pytest.mark.asyncio
async def test_a_refused_dive_is_not_offered(session):
    """The whole point — otherwise it is re-selected hourly forever."""
    await _seed(session, refused_at=REFUSED, label_updated_at=BEFORE)

    assert await _select(session) is None


@pytest.mark.asyncio
async def test_relabelling_brings_it_back(session):
    """A laser label newer than the refusal means the inputs changed.

    This is what keeps the exclusion from outliving the problem. Nobody has to
    clear anything: fixing the labels is the signal.
    """
    await _seed(session, refused_at=REFUSED, label_updated_at=AFTER)

    assert await _select(session) == 1


@pytest.mark.asyncio
async def test_a_label_updated_before_the_refusal_does_not_bring_it_back(session):
    """Guards the comparison direction.

    Inverted, every refusal would be ignored immediately and the fix would do
    nothing — while still looking present in the schema and the code.
    """
    await _seed(session, refused_at=REFUSED, label_updated_at=BEFORE)

    assert await _select(session) is None


@pytest.mark.asyncio
async def test_a_refused_dive_does_not_block_a_healthy_one(session):
    """The failure this exists to prevent, end to end.

    `ORDER BY id LIMIT 1` means a lower-id wedged dive starves every dive
    behind it. Dive 1 is refused, so dive 2 must be what comes back.
    """
    from fishsense_api.models.calibration_target import CalibrationTarget

    session.add(
        CalibrationTarget(
            id=1, name="E4E Checkerboard", rows=10, cols=14, square_size_m=0.042
        )
    )
    refused = _dive(1, calibration_target_id=1)
    refused.calibration_refused_at = REFUSED
    refused.calibration_refused_labels_at = BEFORE
    healthy = _dive(2, calibration_target_id=1)
    session.add_all([refused, healthy])
    await session.flush()
    session.add_all([_image(10, 1), _image(11, 1), _image(20, 2), _image(21, 2)])
    await session.flush()
    session.add_all(
        [
            _laser_label(100, 10, updated_at=BEFORE),
            _laser_label(101, 11, updated_at=BEFORE),
            _laser_label(200, 20, updated_at=BEFORE),
            _laser_label(201, 21, updated_at=BEFORE),
        ]
    )
    await session.flush()

    assert await _select(session) == 2


@pytest.mark.asyncio
async def test_clearing_the_refusal_re_offers_the_dive(session):
    """The operator override, for changes the labels do not capture."""
    from fishsense_api.controllers.dive_controller import clear_calibration_refused

    await _seed(session, refused_at=REFUSED, label_updated_at=BEFORE)
    assert await _select(session) is None

    await clear_calibration_refused(1, session=session)

    assert await _select(session) == 1


@pytest.mark.asyncio
async def test_recording_a_refusal_removes_the_dive(session):
    """The endpoint the fit activities call, driven end to end."""
    from fishsense_api.controllers.dive_controller import set_calibration_refused

    await _seed(session, refused_at=None, label_updated_at=BEFORE)
    assert await _select(session) == 1

    await set_calibration_refused(1, reason="implausible baseline 2.35 cm", session=session)

    assert await _select(session) is None
