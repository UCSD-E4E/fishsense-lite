"""Model representing a model-predicted dive-slate board for an image.

Written by the CPU slate-detector stage (fishsense-core `slate.estimate_plane`)
and read by the dive-slate populate step, which emits it as a Label Studio
pre-annotation so a labeler confirms/nudges the board reference points rather
than placing them from scratch (assisted review). Kept in its own table —
separate from `DiveSlateLabel` — so a prediction never counts toward the
pipeline's slate gates; a labeler's confirmation still lands as a normal
`DiveSlateLabel` via the usual sync.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from sqlmodel import JSON, Column, DateTime, Field, UniqueConstraint

from fishsense_api.models.model_base import ModelBase


class SlatePrediction(ModelBase, table=True):
    """A model-predicted slate board for one image. One row per image —
    re-prediction upserts on the natural key.

    `reference_points` are in rectified-photo pixels (the space labelers place
    `DiveSlateLabel.reference_points` after the composite panel offset is
    stripped), or None when the estimate was rejected. `rejected_reason` records
    why nothing was seeded (`unsupported_slate_family` / `no_board` /
    `low_confidence` / `points_off_canvas`) for seed-rate monitoring.
    """

    __table_args__ = (
        UniqueConstraint("image_id", name="uq_slate_prediction_image"),
    )

    id: int | None = Field(default=None, primary_key=True)
    reference_points: List[List[float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    confidence: float = Field(default=0.0)
    rejected_reason: str | None = Field(default=None)
    # Rectified frame dimensions the points are relative to — the slate populate
    # step needs them to convert pixels to Label Studio keypoint percentages.
    width: int | None = Field(default=None)
    height: int | None = Field(default=None)
    created_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)

    image_id: int | None = Field(default=None, foreign_key="image.id")
