"""SQLModel model for DiveSlate table."""

from datetime import datetime

from pydantic import BaseModel


class DiveSlate(BaseModel):
    """Model representing a dive slate."""

    id: int
    name: str
    dpi: int | None
    path: str
    created_at: datetime | None
