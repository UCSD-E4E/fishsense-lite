"""Per-dive laser-line model for the FishSense API.

The laser dots across a dive's frames are collinear in image space (the
projection of the fixed laser ray), so a RANSAC/TLS fit yields a single 2D
line `a*x + b*y + c = 0` (Hesse normal form, unit normal). Persisting it — a
byproduct the laser-label validation already computes — keeps the within-dive
fit queryable: outlier rejection, and separating laser predictions a labeler
moved from ones they accepted (accepted predictions sit ~1 px off the dive
line, moved ones ~68 px; measured 2026-09-03).

**The line is a WITHIN-DIVE property. It is not a prior for any other dive,
in either direction** (demonstrated 2026-09-03; see CLAUDE.md, "Laser-label
validation"). An earlier version of this docstring called it "the fingerprint
of the mount state" and listed calibration borrow, drift tracking, mount-swap
detection and pooled calibration as things it enables. It does not:

  * Same line ⇏ same laser geometry. The image of a 3D line fixes only the
    plane through the camera centre containing it; where the laser sits
    *within* that plane — the in-plane angle that sets metric scale — moves
    the dots by ~1e-13 px. Dives 383 and 471 agree to 0.22° in the image and
    differ by 3.1° in that angle: +44 % and +305 % length error. Two dives of
    one camera whose lines match can therefore have calibrations that do not
    transfer, and there is no line tolerance that fixes this.
  * Same rig ⇏ same line. The laser can rotate inside its clamp, and the beam
    is off the body axis, so the line moves without the mount being touched.

`line_confidence` / `residual_std` remain a stability signal for the dive
itself — a mount that deformed mid-dive smears the dots off a clean line.
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
