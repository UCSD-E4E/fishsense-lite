"""This module defines the HeadTailLabel model, which represents a head-tail label"""

from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel


class HeadTailLabel(BaseModel):
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
    label_studio_json: Dict[str, Any] | None

    image_id: int | None
    user_id: int | None
