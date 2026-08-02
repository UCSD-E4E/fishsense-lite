"""Module defining the per-dive laser-line fingerprint model for the SDK.

Wire mirror of `fishsense_api.models.dive_laser_line.DiveLaserLine`: the fitted
2D laser line `a*x + b*y + c = 0` (Hesse normal form) plus fit-quality metrics.
See the API model for what the fingerprint is used for.
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
