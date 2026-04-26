"""SQLModel model for DiveSlate table."""

from datetime import datetime
from typing import List, Tuple

from sqlmodel import JSON, Column, DateTime, Field

from fishsense_api.models.model_base import ModelBase


class DiveSlate(ModelBase, table=True):
    """Model representing a dive slate."""

    id: int = Field(default=None, primary_key=True)
    name: str = Field(max_length=100, unique=True, index=True)
    dpi: int | None = Field(default=None)
    path: str = Field(max_length=255, unique=True, index=True)
    created_at: datetime | None = Field(
        sa_type=DateTime(timezone=True), default_factory=datetime.utcnow
    )
    reference_points: List[Tuple[float, float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
