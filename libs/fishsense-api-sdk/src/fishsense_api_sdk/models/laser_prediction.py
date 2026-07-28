"""Module defining laser prediction model for Fishsense API SDK.

Mirrors fishsense-api's `LaserPrediction` SQLModel — a model-predicted laser
dot for an image (fishsense-core LaserDetector stage), in rectified-image
pixels. `x`/`y` are None when no laser was detected; `confidence` is always
set.
"""

from datetime import datetime

from fishsense_api_sdk.models.model_base import ModelBase


class LaserPrediction(ModelBase):
    """Model representing a model-predicted laser dot."""

    id: int | None = None
    x: float | None = None
    y: float | None = None
    confidence: float
    created_at: datetime | None = None
    image_id: int | None = None
