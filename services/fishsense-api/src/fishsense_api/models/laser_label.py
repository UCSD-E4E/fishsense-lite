"""Model representing a laser label from Label Studio."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from sqlmodel import JSON, Column, DateTime, Field, UniqueConstraint

from fishsense_api.models.model_base import ModelBase


class LaserLabel(ModelBase, table=True):
    # pylint: disable=R0801
    """Model representing a laser label."""

    __table_args__ = (
        UniqueConstraint(
            "image_id",
            "label_studio_project_id",
            name="uq_laser_image_project",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    label_studio_task_id: int | None = Field(default=None, unique=True, index=True)
    label_studio_project_id: int | None = Field(default=None, index=True)
    x: float | None = Field(default=None)
    y: float | None = Field(default=None)
    label: str | None = Field(default=None)
    updated_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)
    superseded: bool | None = Field(default=False)
    completed: bool | None = Field(default=False)
    label_studio_json: Dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON)
    )

    image_id: int | None = Field(default=None, foreign_key="image.id")
    user_id: int | None = Field(default=None, foreign_key="user.id")
