"""SQLModel model for DiveSlate table."""

from datetime import datetime
from typing import List, Tuple

from fishsense_api_sdk.models.model_base import ModelBase


class DiveSlate(ModelBase):
    """Model representing a dive slate."""

    id: int
    name: str
    dpi: int | None
    path: str
    created_at: datetime | None
    reference_points: List[Tuple[float, float]] | None
