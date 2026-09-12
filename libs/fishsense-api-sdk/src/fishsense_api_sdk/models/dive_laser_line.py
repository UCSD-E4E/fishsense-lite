"""Module defining the per-dive laser-line model for the SDK.

Wire mirror of `fishsense_api.models.dive_laser_line.DiveLaserLine`: the fitted
2D laser line `a*x + b*y + c = 0` (Hesse normal form) plus fit-quality metrics.

The line is a within-dive property (outlier rejection, moved-prediction
detection). It is never a prior for another dive: two dives whose lines agree
can differ by degrees in the in-plane laser angle that sets metric scale, which
the line cannot see. See the API model's module docstring for the measured
case (dives 383/471) before using it to rank calibration borrows.
"""

from datetime import datetime

from fishsense_api_sdk.models.model_base import ModelBase


class DiveLaserLine(ModelBase):
    """A dive's fitted 2D laser line plus fit-quality metrics."""

    id: int | None = None
    dive_id: int | None = None

    a: float
    b: float
    c: float

    n_points: int
    inlier_count: int
    inlier_fraction: float
    residual_std: float
    label_noise_mad: float
    line_confidence: float

    fitted_at: datetime | None = None
