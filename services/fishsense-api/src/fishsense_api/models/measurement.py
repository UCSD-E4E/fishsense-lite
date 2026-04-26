"""Measurement model for the FishSense API."""

from sqlmodel import Field, SQLModel


class Measurement(SQLModel, table=True):
    """Measurement model representing fish measurements in the database."""

    id: int | None = Field(default=None, primary_key=True)
    length_m: float | None = Field(default=None)

    image_id: int | None = Field(default=None, foreign_key="image.id")
    fish_id: int | None = Field(default=None, foreign_key="fish.id")
