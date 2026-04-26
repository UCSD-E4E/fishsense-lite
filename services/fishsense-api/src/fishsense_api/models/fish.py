"""Fish model for the FishSense API."""

from sqlmodel import Field, SQLModel


class Fish(SQLModel, table=True):
    """Fish model representing a fish in the database."""

    id: int | None = Field(default=None, primary_key=True)

    species_id: int | None = Field(default=None, foreign_key="species.id")
