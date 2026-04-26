"""Model representing a user."""

from datetime import datetime

from sqlmodel import DateTime, Field, UniqueConstraint

from fishsense_api.models.model_base import ModelBase


class User(ModelBase, table=True):
    """Model representing a user."""

    __table_args__ = (
        UniqueConstraint(
            "label_studio_id",
            name="uq_user_label_studio_id",
        ),
        UniqueConstraint(
            "email",
            name="uq_user_email",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    label_studio_id: int | None = Field(default=None, unique=True, index=True)
    email: str | None = Field(max_length=100, unique=True, index=True)
    first_name: str | None = Field(max_length=100)
    last_name: str | None = Field(max_length=100)
    last_activity: datetime | None = Field(
        sa_type=DateTime(timezone=True), default=None
    )
    date_joined: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)
