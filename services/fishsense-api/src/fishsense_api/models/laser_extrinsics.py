"""Laser extrinsics model for the FishSense API."""

from datetime import datetime
from typing import List

from sqlalchemy import UniqueConstraint, text
from sqlmodel import JSON, Column, DateTime, Field

from fishsense_api.models.model_base import ModelBase


class LaserExtrinsics(ModelBase, table=True):
    """Laser extrinsics model representing laser calibration data in the database.

    One row per dive (`uq_laserextrinsics_dive_id`): calibration is a per-dive
    property, `put_laser_extrinsics_for_dive` upserts on `dive_id`, and no
    consumer reads a history of extrinsics — reads take the single row (or borrow
    a sibling's via `Dive.calibration_dive_id`). `created_at` carries a
    `server_default=now()` so a row can never be inserted with a NULL timestamp
    (which broke the latest-wins read: Postgres sorts NULLs first under DESC).
    """

    __table_args__ = (UniqueConstraint("dive_id", name="uq_laserextrinsics_dive_id"),)

    id: int | None = Field(default=None, primary_key=True)
    laser_position: List[float] = Field(default_factory=list, sa_column=Column(JSON))
    laser_axis: List[float] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime | None = Field(
        sa_type=DateTime(timezone=True),
        default=None,
        sa_column_kwargs={"server_default": text("now()")},
    )

    dive_id: int | None = Field(default=None, foreign_key="dive.id")
    camera_id: int = Field(default=None, foreign_key="camera.id")
