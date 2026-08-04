"""Fish model for FishSense API SDK."""

from pydantic import BaseModel


class Fish(BaseModel):
    """Fish model representing fish information."""

    id: int | None

    # Natural key for a physical fish model; NULL for real fish. Mirrors the
    # API SQLModel's nullable-unique `name`.
    name: str | None = None

    species_id: int | None
