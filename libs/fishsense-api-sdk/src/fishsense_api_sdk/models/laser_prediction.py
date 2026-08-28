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
    # Rectified frame dimensions the x/y are relative to (for the pixel ->
    # Label Studio keypoint-percentage conversion in laser populate).
    width: int | None = None
    height: int | None = None
    # Laser colour read off the dot's pixels ("red" / "green"), signed R-G at
    # the dot in 8-bit levels, and whether a detected dot was dropped for
    # falling outside the expected-laser region. See the API SQLModel.
    color: str | None = None
    color_margin: float | None = None
    rejected_out_of_region: bool = False
    created_at: datetime | None = None
    image_id: int | None = None
