"""Species Label Model for FishSense API."""

from datetime import datetime
from typing import Any, Dict

from sqlmodel import JSON, Column, DateTime, Field, UniqueConstraint

from fishsense_api.models.model_base import ModelBase


class SpeciesLabel(ModelBase, table=True):
    # pylint: disable=R0801
    """Model representing a species label."""

    __table_args__ = (
        UniqueConstraint(
            "image_id",
            "label_studio_project_id",
            name="uq_species_image_project",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    label_studio_task_id: int | None = Field(default=None, unique=True, index=True)
    label_studio_project_id: int | None = Field(default=None, index=True)
    image_url: str | None = Field(default=None)
    updated_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)
    completed: bool | None = Field(default=False)
    grouping: str | None = Field(default=None)
    top_three_photos_of_group: bool | None = Field(default=None)
    slate_upside_down: bool | None = Field(default=None)
    laser_x: float | None = Field(default=None)
    laser_y: float | None = Field(default=None)
    laser_label: str | None = Field(default=None)
    content_of_image: str | None = Field(default=None)
    fish_measurable_category: str | None = Field(default=None)
    fish_angle_category: str | None = Field(default=None)
    fish_curved_category: str | None = Field(default=None)
    label_studio_json: Dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON)
    )

    image_id: int | None = Field(default=None, foreign_key="image.id")
    user_id: int | None = Field(default=None, foreign_key="user.id")
