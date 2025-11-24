"""Model for Slate Labels."""

from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel


class DiveSlateLabels(BaseModel):
    """Model representing slate labels."""

    id: int | None
    label_studio_task_id: int | None
    label_studio_project_id: int | None
    image_url: str | None
    updated_at: datetime | None
    completed: bool | None
    label_studio_json: Dict[str, Any] | None

    image_id: int | None
    user_id: int | None
