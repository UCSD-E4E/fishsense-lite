"""Module defining species label model for Fishsense API SDK."""

from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel


class SpeciesLabel(BaseModel):
    # pylint: disable=R0801
    """Model representing a species label."""

    id: int | None
    label_studio_task_id: int | None
    label_studio_project_id: int | None
    image_url: str | None
    updated_at: datetime | None
    completed: bool | None
    grouping: str | None
    top_three_photos_of_group: bool | None
    slate_upside_down: bool | None
    laser_x: float | None
    laser_y: float | None
    laser_label: str | None
    content_of_image: str | None
    fish_measurable_category: str | None
    fish_angle_category: str | None
    fish_curved_category: str | None
    label_studio_json: Dict[str, Any] | None

    image_id: int | None
    user_id: int | None
