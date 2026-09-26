"""Module defining laser label model for Fishsense API SDK."""

from datetime import datetime
from typing import Any, Dict

from fishsense_api_sdk.models.model_base import ModelBase
from fishsense_api_sdk.models.superseded_reason import SupersededReason


class LaserLabel(ModelBase):
    """Model representing a laser label."""

    id: int | None
    label_studio_task_id: int | None
    label_studio_project_id: int | None
    x: float | None
    y: float | None
    label: str | None
    updated_at: datetime | None
    superseded: bool | None
    # Who superseded it; None = unknown. Set it whenever you set `superseded`.
    superseded_reason: SupersededReason | None = None
    completed: bool | None
    needs_reprocess: bool = False
    label_studio_json: Dict[str, Any] | str | None

    image_id: int | None
    user_id: int | None
