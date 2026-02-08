"""Model for Slate Labels."""

from datetime import datetime
from typing import Any, Dict, List, Tuple

from fishsense_api_sdk.models.model_base import ModelBase


class DiveSlateLabel(ModelBase):
    # pylint: disable=R0801
    """Model representing slate labels."""

    id: int | None
    label_studio_task_id: int | None
    label_studio_project_id: int | None
    image_url: str | None
    upside_down: bool | None
    reference_points: List[Tuple[float, float]] | None
    slate_rectangle: List[Tuple[float, float]] | None
    skipped_points: List[int] | None
    updated_at: datetime | None
    completed: bool | None
    label_studio_json: Dict[str, Any] | None

    image_id: int | None
    user_id: int | None
