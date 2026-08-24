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
    created_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)

    image_id: int | None = Field(default=None, foreign_key="image.id")
