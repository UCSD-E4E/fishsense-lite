"""What a physically plausible laser baseline is, spelled once.

The baseline is `norm(laser_position[:2])` — the offset between the camera
centre and the laser on one rig. It is hardware, not a property of a dive, and
measurement says so loudly: over all 35 stored calibrations its interquartile
range is **9.99-10.45 cm**, half a centimetre across both producers, every
camera and two years.

It lives in `fishsense_shared` for the same reason `taxonomy` and
`task_queues` do — two services need the identical number and neither owns it:

* the **data-worker** refuses to persist a fit outside these bounds
  (`calibration_consistency.check_baseline_plausible`);
* the **api** excludes an already-stored calibration outside them from
  counting as a calibration at all, so its dive re-enters the calibration
  cohorts instead of being measured against a fit we know is wrong.

A copy in each would be the drift this repo keeps rediscovering, and here the
two halves disagreeing is especially quiet: the api would hand a dive back for
recalibration that the data-worker then persists unchanged, hourly, forever.

**Why bounding the answer is the only check that works.** Nothing upstream
distinguishes a good fit from a bad one. Measured 2026-09-11 by rendering the
detected lattices: 60 of 61 frames found the full 14x10 board on good and bad
dives alike. Observation count does not separate them (one dive fits 2.60 cm
from 19 clean observations, three fit correctly from 2), nor does depth spread,
nor detection rate. And `check_fit_self_consistency` is structurally blind to
the baseline: it compares the fitted ray's *projection* to its 2-D dots, and
sliding the laser's offset leaves that projection identical.
"""

from __future__ import annotations

import math

__all__ = [
    "MAX_BASELINE_M",
    "MIN_BASELINE_M",
    "baseline_m",
    "is_plausible_baseline",
]

#: Bounds on the baseline, in metres.
#:
#: **Placed midway between the populations, not hard against the healthy one.**
#: Healthy extremes are 8.90 and 12.95 cm; the nearest bad fits are 6.91 and
#: 16.01 cm. A first draft used 8-13 cm, which left half a millimetre of
#: headroom above the widest sound calibration while sitting 3 cm clear of the
#: nearest bad one -- so ordinary variation or a 2% pitch error would have been
#: refused, and a refusal is expensive (it wedges the dive in its cohort).
#:
#: **Widen only against re-measured data.** A wrong baseline is the one error
#: the rest of the pipeline provably cannot see: it scales every depth, hence
#: every length, while reprojection residual and self-consistency stay clean.
MIN_BASELINE_M = 0.078
MAX_BASELINE_M = 0.145


def baseline_m(laser_position) -> float:
    """The in-plane offset of `laser_position`, in metres.

    Only x and y are read. Both producers return the fit's origin as the point
    where the laser ray crosses the camera's z=0 plane and pad z to zero, so a
    three-component norm would depend on padding that carries no information.

    Returns `inf` for anything unreadable — a short vector, a non-numeric
    entry — so callers get the same answer they would for an absurd baseline
    rather than an exception. "Cannot tell" and "implausible" want the same
    treatment here: refuse, and keep looking.
    """
    try:
        x = float(laser_position[0])
        y = float(laser_position[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return float("inf")
    magnitude = (x * x + y * y) ** 0.5
    # NaN fails every comparison, so a caller's range test would ACCEPT it.
    # Map it onto the same answer as unreadable input. `math.isnan` rather than
    # `magnitude != magnitude`, which reads as a typo and pylint flags.
    if math.isnan(magnitude):
        return float("inf")
    return magnitude


def is_plausible_baseline(
    laser_position,
    *,
    min_baseline_m: float = MIN_BASELINE_M,
    max_baseline_m: float = MAX_BASELINE_M,
) -> bool:
    """Whether `laser_position` describes a baseline a real rig could have."""
    return min_baseline_m <= baseline_m(laser_position) <= max_baseline_m
