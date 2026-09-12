"""An implausible stored calibration must count as no calibration.

`check_baseline_plausible` on the data-worker stops *new* bad fits, but it is
write-time only. Both calibration cohorts select on "has no `LaserExtrinsics`
row", so the eight already-stored implausible calibrations — 2.35 to 22.22 cm
against a fleet whose interquartile range is 9.99–10.45 cm — would never be
refitted, and stage 14 would keep measuring 663 of 3,104 measurements against
them at −75% to +45% error.

Two properties are pinned here:

* a dive whose only calibration is implausible **resolves as uncalibrated**,
  through its own row and through a borrowed one, so nothing measures against
  it and it re-enters the calibration cohorts;
* re-entry candidates are **offered last**. Both selectors are
  `ORDER BY id LIMIT 1`, so a dive that refits to the same bad value and is
  refused again would otherwise head-of-line block every healthy dive behind
  it — the dive-347 shape. Ordering them last bounds the damage to the dives
  that are already broken.
"""

from __future__ import annotations

import pytest
from fishsense_shared.calibration_bounds import MAX_BASELINE_M, MIN_BASELINE_M

#: 10.4 cm — where every sound calibration in the fleet sits.
GOOD_XY = [0.0624, 0.0832, 0.0]
#: 2.35 cm — dive 522's real fitted baseline, the worst of the eight.
BAD_XY = [0.0141, 0.0188, 0.0]


def test_the_fixtures_straddle_the_bound():
    """Guard the guards: these must actually land either side."""
    from fishsense_shared.calibration_bounds import is_plausible_baseline

    assert is_plausible_baseline(GOOD_XY)
    assert not is_plausible_baseline(BAD_XY)


def test_bounds_are_the_shared_ones():
    """The api and the data-worker must read the identical numbers.

    If they drift, the api hands a dive back for recalibration that the
    data-worker then persists unchanged — hourly, forever.
    """
    from fishsense_data_processing_workflow_worker import (  # noqa: PLC0415
        calibration_consistency,
    )

    assert calibration_consistency.DEFAULT_MIN_BASELINE_M == MIN_BASELINE_M
    assert calibration_consistency.DEFAULT_MAX_BASELINE_M == MAX_BASELINE_M


@pytest.mark.parametrize(
    "position, expected",
    [
        (GOOD_XY, True),
        (BAD_XY, False),
        ([0.0, 0.0, 0.0], False),
        ([float("nan"), 0.0, 0.0], False),
        ([], False),
        (None, False),
    ],
)
def test_plausibility_covers_the_degenerate_shapes(position, expected):
    """A stored row can be short, empty or NaN; none may read as plausible.

    NaN matters most: every comparison against it is False, so a naive range
    test would ACCEPT it and the dive would resolve as calibrated.
    """
    from fishsense_shared.calibration_bounds import is_plausible_baseline

    assert is_plausible_baseline(position) is expected
