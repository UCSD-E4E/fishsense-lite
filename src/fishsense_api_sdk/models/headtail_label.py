"""This module defines the HeadTailLabel model, which represents a head-tail label"""

from datetime import datetime

from fishsense_api_sdk.models.label_studio_json import LabelStudioJson
from fishsense_api_sdk.models.model_base import ModelBase


class HeadTailLabel(ModelBase):
    # pylint: disable=R0801
    """Model representing a head-tail label."""

    id: int | None
    label_studio_task_id: int | None
    label_studio_project_id: int | None
    head_x: float | None
    head_y: float | None
    tail_x: float | None
    tail_y: float | None
    updated_at: datetime | None
    superseded: bool | None
    completed: bool | None
    label_studio_json: LabelStudioJson | None

    image_id: int | None
    user_id: int | None
