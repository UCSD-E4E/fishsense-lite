"""Module defining dive model for Fishsense API SDK."""

from datetime import datetime

from pydantic import BaseModel

from fishsense_api_sdk.models.priority import Priority


class Dive(BaseModel):
    """Model representing a dive."""

    id: int | None
    name: str | None
    path: str
    dive_datetime: datetime
    priority: Priority
    flip_dive_slate: bool | None

    camera_id: int | None
    dive_slate_id: int | None
