"""Species model for FishSense API SDK."""

from pydantic import BaseModel


class Species(BaseModel):
    """Species model representing fish species information."""

    id: int | None
    scientific_name: str | None
    common_name: str | None
