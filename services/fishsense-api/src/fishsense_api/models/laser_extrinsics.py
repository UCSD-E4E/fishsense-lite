"""Laser extrinsics model for the FishSense API."""

from datetime import datetime
from typing import List

from sqlmodel import JSON, Column, DateTime, Field

from fishsense_api.models.model_base import ModelBase


class LaserExtrinsics(ModelBase, table=True):
    """Laser extrinsics model representing laser calibration data in the database."""

    id: int | None = Field(default=None, primary_key=True)
    laser_position: List[float] = Field(default_factory=list, sa_column=Column(JSON))
    laser_axis: List[float] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)

    dive_id: int | None = Field(default=None, foreign_key="dive.id")
    camera_id: int = Field(default=None, foreign_key="camera.id")
