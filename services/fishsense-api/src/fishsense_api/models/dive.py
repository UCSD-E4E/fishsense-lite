"""Model representing a dive."""

from datetime import datetime

from sqlmodel import Column, DateTime, Enum, Field

from fishsense_api.models.model_base import ModelBase
from fishsense_api.models.priority import Priority


class Dive(ModelBase, table=True):
    """Model representing a dive."""

    id: int | None = Field(default=None, primary_key=True)
    name: str | None = Field(default=None, index=True)
    path: str = Field(max_length=255, unique=True, index=True)
    dive_datetime: datetime = Field(sa_type=DateTime(timezone=True), default=None)
    priority: Priority = Field(default=Priority.LOW, sa_column=Column(Enum(Priority)))
    flip_dive_slate: bool | None = Field(default=False)

    camera_id: int | None = Field(default=None, foreign_key="camera.id")
    dive_slate_id: int | None = Field(default=None, foreign_key="diveslate.id")
