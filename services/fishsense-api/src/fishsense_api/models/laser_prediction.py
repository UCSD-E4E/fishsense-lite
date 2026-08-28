"""Model representing a model-predicted laser dot for an image.

Written by the GPU laser-detector stage (fishsense-core `LaserDetector`) and
read by the laser populate step, which emits it as a Label Studio
pre-annotation so a labeler confirms/nudges rather than placing from scratch
(assisted review). Kept in its own table — separate from `LaserLabel` — so a
prediction never counts toward the human "valid laser" gates; a labeler's
confirmation still lands as a normal `LaserLabel` via the usual sync.
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import DateTime, Field, UniqueConstraint

from fishsense_api.models.model_base import ModelBase


class LaserPrediction(ModelBase, table=True):
    """A model-predicted laser dot, in rectified-image pixels (the space
    labelers place `LaserLabel.x/y`). One row per image — re-prediction
    upserts on the natural key."""

    __table_args__ = (UniqueConstraint("image_id", name="uq_laser_prediction_image"),)

    id: int | None = Field(default=None, primary_key=True)
    # x/y are None when the detector found no laser; confidence is always set.
    x: float | None = Field(default=None)
    y: float | None = Field(default=None)
    confidence: float = Field(default=0.0)
    # Rectified frame dimensions the x/y are relative to — the laser populate
    # step needs them to convert pixels to Label Studio keypoint percentages.
    width: int | None = Field(default=None)
    height: int | None = Field(default=None)
    # Laser colour read off the dot's own pixels: "red" / "green", or NULL
    # when there was no dot to sample or the channels were too close to call.
    # Advisory per row — populate takes the dive-level majority, because
    # colour is a property of the rig for the whole dive (143 prod dives are
    # entirely red, 88 entirely green) and a single frame is ~98% reliable.
    color: str | None = Field(default=None)
    # Signed R-G at the dot in 8-bit levels, positive for red. Recorded, not
    # gated on: it is what makes a close call auditable after the fact, the
    # same role `LaserDepth.residual_m` plays for the depth stage.
    color_margin: float | None = Field(default=None)
    # The detector found a dot, but outside the expected-laser region, so x/y
    # were dropped. Distinct from an ordinary non-detection (the model found
    # nothing): without this the two are indistinguishable, and a mis-sized
    # region would read as a model that had stopped working.
    rejected_out_of_region: bool = Field(default=False)
    created_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)

    image_id: int | None = Field(default=None, foreign_key="image.id")
