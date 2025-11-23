"""SQLModel model for DiveSlate table."""

from datetime import datetime
from typing import List, Tuple

from pydantic import BaseModel


class DiveSlate(BaseModel):
    """Model representing a dive slate."""

    id: int
    name: str
    dpi: int | None
    path: str
    created_at: datetime | None
    points_json: List[Tuple[float, float]] | None
