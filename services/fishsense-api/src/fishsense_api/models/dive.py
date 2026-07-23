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

    # Self-referential link to the dive whose laser calibration this dive
    # borrows. Laser calibration is physically a property of the camera+laser
    # rig, not the dive, so a dive with no slate frames of its own (e.g. a
    # fish-only dive) can point at a sibling slate/calibration dive shot with
    # the same rig. When set, laser-extrinsics resolution and the
    # `calibrated` gate fall back to this dive's LaserExtrinsics. NULL means
    # "self-calibrate from my own slate labels" (the default).
    calibration_dive_id: int | None = Field(default=None, foreign_key="dive.id")
