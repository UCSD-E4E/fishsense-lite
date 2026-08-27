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

    # Free-text operator note. Exists to carry the *reason* a dive is in an
    # unusual state — overwhelmingly, why it was set `Priority.NONE`. Nothing
    # in the pipeline reads it: it is for the human looking at a dive that
    # will never drain and asking why. Deliberately unstructured, because the
    # reasons are one-offs (a slate with no scan, a corrupted card, a dive
    # shot on a rig whose calibration was never recoverable) and any schema
    # we invented for them would be wrong by the third case.
    notes: str | None = Field(default=None)

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
