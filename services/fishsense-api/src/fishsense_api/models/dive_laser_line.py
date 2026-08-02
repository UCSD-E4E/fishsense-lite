"""Per-dive laser-line fingerprint model for the FishSense API.

The laser dots across a dive's frames are collinear in image space (the
projection of the fixed laser ray), so a RANSAC/TLS fit yields a single 2D
line `a*x + b*y + c = 0` (Hesse normal form, unit normal). That line is the
*fingerprint of the mount state*: on a given camera it changes only when the
cold-shoe mount rotates, drifts (PLA thermal/creep), or is swapped. Persisting
it — a byproduct the laser-label validation already computes — turns four
questions into queries over `(camera_id, line)`:

  * borrow: two dives on one camera with the same confident line share a 3D
    laser geometry, so one's `LaserExtrinsics` transfers to the other;
  * drift: the line's wander over time per camera;
  * mount swaps: step discontinuities in that per-camera series;
  * pooled calibration: dives with a matching line can co-fit one calibration.

`line_confidence` / `residual_std` double as a stability signal — a mount that
deformed mid-dive smears the dots off a clean line and shows up as a poor fit.
"""

from datetime import datetime

from sqlalchemy import UniqueConstraint, func
from sqlmodel import DateTime, Field

from fishsense_api.models.model_base import ModelBase


class DiveLaserLine(ModelBase, table=True):
    """A dive's fitted 2D laser line `a*x + b*y + c = 0` plus fit-quality metrics.

    One row per dive (`uq_diveslaserline_dive_id`); `put_dive_laser_line`
    upserts on `dive_id`. `camera_id` / `dive_datetime` are NOT duplicated here
    — consumers join `Dive`. Same NULL-safe `server_default` pattern as
    LaserExtrinsics so a row can never be inserted with a NULL timestamp.
    """

    __table_args__ = (UniqueConstraint("dive_id", name="uq_divelaserline_dive_id"),)

    id: int | None = Field(default=None, primary_key=True)
    dive_id: int | None = Field(default=None, foreign_key="dive.id")

    # Line in Hesse normal form: a*x + b*y + c = 0 with a^2 + b^2 = 1.
    a: float
    b: float
    c: float

    # Fit-quality metrics (mirror line_fit.LineFit); double as a stability signal.
    n_points: int
    inlier_count: int
    inlier_fraction: float
    residual_std: float
    label_noise_mad: float
    line_confidence: float

    fitted_at: datetime | None = Field(
        sa_type=DateTime(timezone=True),
        default=None,
        # func.now() is dialect-aware (CURRENT_TIMESTAMP on sqlite, now() on
        # Postgres); pylint's not-callable on func.* is a known false positive.
        sa_column_kwargs={"server_default": func.now()},  # pylint: disable=not-callable
    )
