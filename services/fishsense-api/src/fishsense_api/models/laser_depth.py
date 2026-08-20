"""Per-image distance to the laser dot.

Derived data, not a label: the data-worker projects the laser label's pixel
through `WorldPointHandler.compute_world_point_from_laser` against the dive's
resolved `LaserExtrinsics` and the camera's intrinsics, and records where the
dot actually was in space. Stage 14 has always computed this en route to a
fish length and thrown it away, so it existed for measurable frames only and
was never queryable; this table keeps it for **every** image carrying a valid
laser label.

Two distances, because they are not the same number and the difference is
easy to get wrong. `depth_m` is the Z component — distance along the optical
axis, which is what stage 14 back-projects head and tail against — and
`range_m` is the Euclidean norm, the true slant distance to the dot. They
diverge with the dot's off-axis angle.

The row names the label and the calibration it came from. Both are inputs, so
either changing invalidates the depth: a relabel moves the dot, and a
recalibration moves the ray it is projected against (the 2026-08-11 slate
panel-offset fix recalibrated 6 of 8 dives that already had measurements).
`select_next_for_laser_depth` re-picks a dive on exactly that mismatch, which
is what makes a recompute self-healing rather than a manual sweep.
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import DateTime, Field, UniqueConstraint

from fishsense_api.models.model_base import ModelBase


class LaserDepth(ModelBase, table=True):
    """The distance to an image's laser dot, in metres. One row per image —
    a recompute upserts on the natural key."""

    __table_args__ = (UniqueConstraint("image_id", name="uq_laser_depth_image"),)

    id: int | None = Field(default=None, primary_key=True)
    # Along the optical axis (laser3d[2]) — the depth stage 14 measures at.
    depth_m: float = Field()
    # Euclidean |laser3d| — the true distance to the dot. Stored alongside so
    # a consumer never has to guess which one `depth_m` meant.
    range_m: float | None = Field(default=None)
    # How close the camera ray and the laser ray actually came to meeting, in
    # metres. ~0 when the dot is genuinely consistent with the calibration, so
    # it is a direct per-dot check of a laser label against its extrinsics.
    #
    # Recorded, deliberately not gated on. It is necessary but not sufficient:
    # blind to error *along* the laser's epipolar line (a dot slid along it
    # moves the depth while the residual stays ~0), zero for two rays that
    # meet at the camera centre, and metric — so one fixed threshold is
    # stricter close up than far away. The float32 solve also puts the noise
    # floor around 1e-5 m at metre scale. A threshold should come from the
    # observed distribution, not from a guess.
    residual_m: float | None = Field(default=None)
    created_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)

    image_id: int | None = Field(default=None, foreign_key="image.id")
    # Provenance of the two inputs. Nullable because the FKs are nullable
    # everywhere else in this schema, but the compute activity always sets
    # both — a row without them can never be recognised as current and would
    # be recomputed on the next sweep.
    laser_label_id: int | None = Field(default=None, foreign_key="laserlabel.id")
    laser_extrinsics_id: int | None = Field(
        default=None, foreign_key="laserextrinsics.id"
    )
