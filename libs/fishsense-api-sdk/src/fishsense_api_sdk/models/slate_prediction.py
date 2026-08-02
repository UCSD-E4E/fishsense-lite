"""Module defining slate prediction model for Fishsense API SDK.

Mirrors fishsense-api's `SlatePrediction` SQLModel — a model-predicted dive-slate
board for an image (fishsense-core slate-detector stage). `reference_points` are
in rectified-photo pixels, or None when the estimate was rejected (see
`rejected_reason`); `confidence` is always set.
"""

from datetime import datetime
from typing import List

from fishsense_api_sdk.models.model_base import ModelBase


class SlatePrediction(ModelBase):
    """Model representing a model-predicted dive-slate board."""

    id: int | None = None
    reference_points: List[List[float]] | None = None
    confidence: float
    rejected_reason: str | None = None
    # Rectified frame dimensions the points are relative to (for the pixel ->
    # Label Studio keypoint-percentage conversion in slate populate).
    width: int | None = None
    height: int | None = None
    created_at: datetime | None = None
    image_id: int | None = None
