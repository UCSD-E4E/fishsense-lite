"""Fish model for FishSense API SDK."""

from pydantic import BaseModel


class Fish(BaseModel):
    """Fish model representing fish information."""

    id: int | None

    species_id: int | None
