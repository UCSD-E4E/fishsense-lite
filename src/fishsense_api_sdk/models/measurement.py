"""Measurement model for FishSense API SDK."""

from pydantic import BaseModel


class Measurement(BaseModel):
    """Measurement model representing fish measurement information."""

    id: int | None
    length_m: float | None

    image_id: int | None
    fish_id: int | None
