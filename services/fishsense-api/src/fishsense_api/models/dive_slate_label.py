"""Model for Slate Labels."""

from datetime import datetime
from typing import Any, Dict, List, Tuple

from sqlmodel import JSON, Column, DateTime, Field, UniqueConstraint

from fishsense_api.models.model_base import ModelBase


class DiveSlateLabel(ModelBase, table=True):
    # pylint: disable=R0801
    """Model representing slate labels."""

    __table_args__ = (
        UniqueConstraint(
            "image_id",
            "label_studio_project_id",
            name="uq_dive_slate_image_project",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    label_studio_task_id: int | None = Field(default=None, unique=True, index=True)
    label_studio_project_id: int | None = Field(default=None, index=True)
    upside_down: bool | None = Field(default=None)
    reference_points: List[Tuple[float, float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    slate_rectangle: List[Tuple[float, float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    skipped_points: List[int] | None = Field(default=None, sa_column=Column(JSON))
    image_url: str | None = Field(default=None)
    updated_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)
    completed: bool | None = Field(default=False)
    label_studio_json: Dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON)
    )

    image_id: int | None = Field(default=None, foreign_key="image.id")
    user_id: int | None = Field(default=None, foreign_key="user.id")
