"""Module defining dive model for Fishsense API SDK."""

from datetime import datetime

from fishsense_api_sdk.models.model_base import ModelBase
from fishsense_api_sdk.models.priority import Priority


class Dive(ModelBase):
    """Model representing a dive."""

    id: int | None
    name: str | None
    path: str
    dive_datetime: datetime
    priority: Priority
    flip_dive_slate: bool | None

    camera_id: int | None
    dive_slate_id: int | None
    # Defaulted (unlike the fields above) so consumers built before this
    # column existed — older API responses, worker test fixtures — still
    # validate. A newly-added optional column must be optional on the wire.
    calibration_dive_id: int | None = None
