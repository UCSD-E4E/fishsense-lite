"""Module defining image model for Fishsense API SDK."""

from datetime import datetime

from fishsense_api_sdk.models.model_base import ModelBase


class Image(ModelBase):
    """Model representing an image."""

    id: int | None
    path: str
    taken_datetime: datetime
    checksum: str
    is_canonical: bool

    dive_id: int | None
    camera_id: int | None
