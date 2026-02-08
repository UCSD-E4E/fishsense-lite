"""Module defining laser label model for Fishsense API SDK."""

from datetime import datetime

from fishsense_api_sdk.models.label_studio_json import LabelStudioJson
from fishsense_api_sdk.models.model_base import ModelBase


class LaserLabel(ModelBase):
    # pylint: disable=R0801
    """Model representing a laser label."""

    id: int | None
    label_studio_task_id: int | None
    label_studio_project_id: int | None
    x: float | None
    y: float | None
    label: str | None
    updated_at: datetime | None
    superseded: bool | None
    completed: bool | None
    label_studio_json: LabelStudioJson | None

    image_id: int | None
    user_id: int | None
