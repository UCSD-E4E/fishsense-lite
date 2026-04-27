"""Model representing a user in the Fishsense API SDK."""

from datetime import datetime

from fishsense_api_sdk.models.model_base import ModelBase


class User(ModelBase):
    """Model representing a user."""

    id: int | None
    label_studio_id: int | None
    email: str | None
    first_name: str | None
    last_name: str | None
    last_activity: datetime | None
    date_joined: datetime | None
