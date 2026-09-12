"""Unit tests for the shared laser-baseline bounds.

The baseline is the offset between camera and laser on one rig — hardware, not
a property of a dive. Over all 35 stored calibrations its interquartile range
is 9.99-10.45 cm, half a centimetre across both producers, every camera and two
years. Eight fits sit outside 7.8-14.5 cm and back 663 of 3,104 measurements at
-75% to +45% error.

Two services read these numbers: the data-worker refuses to persist a fit
outside them, and the api stops an already-stored one outside them from
counting as a calibration at all. Both behaviours are tested in their own
packages; what is pinned here is the arithmetic they share.
"""

from __future__ import annotations

import math

import pytest

from fishsense_shared.calibration_bounds import (
    MAX_BASELINE_M,
    MIN_BASELINE_M,
    baseline_m,
    is_plausible_baseline,
)


def test_baseline_is_the_in_plane_norm():
    assert baseline_m([0.06, 0.08, 0.0]) == pytest.approx(0.10)


def test_the_z_component_is_ignored():
    """Both producers pad z to zero on an origin that is the z=0 crossing.

    Reading a three-component norm would make the answer depend on padding
    that carries no information, and would drift if that convention changed.
    """
    assert baseline_m([0.06, 0.08, 5.0]) == pytest.approx(0.10)


@pytest.mark.parametrize(
    "position",
    [
        [],
        [0.06],
        None,
        "not a position",
        [None, None],
        ["a", "b"],
        {"x": 1},
    ],
)
def test_unreadable_positions_report_infinite(position):
    """"Cannot tell" and "implausible" want the same treatment: refuse.

    Returning `inf` rather than raising means a caller filtering a table of
    stored rows cannot be derailed by one malformed entry.
    """
    assert baseline_m(position) == math.inf


def test_nan_reports_infinite_rather_than_nan():
    """NaN fails every comparison, so a caller's range test would ACCEPT it."""
    assert baseline_m([float("nan"), 0.0]) == math.inf


def test_infinite_input_is_not_plausible():
    assert not is_plausible_baseline([float("inf"), 0.0])


@pytest.mark.parametrize("baseline_cm", [8.90, 9.51, 10.12, 10.36, 10.45, 12.95])
def test_accepts_the_baselines_the_fleet_actually_produces(baseline_cm):
    """Measured values from calibrations whose lengths grade within -12.7%."""
    assert is_plausible_baseline([baseline_cm / 100, 0.0, 0.0])


@pytest.mark.parametrize(
    "baseline_cm", [2.35, 2.60, 4.81, 4.90, 6.25, 6.91, 16.01, 22.22]
)
def test_refuses_the_baselines_that_produced_bad_lengths(baseline_cm):
    """The eight known-bad fits, by their real measured baselines.

    Parametrised on the actual numbers so that widening the bounds far enough
    to re-admit one fails the build.
    """
    assert not is_plausible_baseline([baseline_cm / 100, 0.0, 0.0])


def test_a_zero_baseline_is_not_plausible():
    """A laser at the camera centre is not a rig, it is a failed fit."""
    assert not is_plausible_baseline([0.0, 0.0, 0.0])


def test_the_bounds_are_inclusive_at_the_edges():
    assert is_plausible_baseline([MIN_BASELINE_M, 0.0])
    assert is_plausible_baseline([MAX_BASELINE_M, 0.0])


def test_overrides_are_honoured():
    """So a caller can tighten for an experiment without editing the module."""
    assert not is_plausible_baseline(
        [0.104, 0.0], min_baseline_m=0.11, max_baseline_m=0.12
    )


def test_bounds_bracket_the_measured_cluster():
    """The constants themselves, so widening them is a deliberate act.

    The fleet sits at 9.99-10.45 cm interquartile. The bounds must contain that
    comfortably while still excluding 6.91 and 16.01 cm, the nearest bad fits
    on each side — and sit nearer the midpoint than hard against the healthy
    extremes, or ordinary variation gets refused.
    """
    assert MIN_BASELINE_M < 0.0890
    assert MIN_BASELINE_M > 0.0691
    assert MAX_BASELINE_M > 0.1295
    assert MAX_BASELINE_M < 0.1601
