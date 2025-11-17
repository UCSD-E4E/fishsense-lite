"""Model representing a user in the Fishsense API SDK."""

from pydantic import AwareDatetime, BaseModel


class User(BaseModel):
    """Model representing a user."""

    id: int | None
    label_studio_id: int | None
    email: str | None
    first_name: str | None
    last_name: str | None
    last_activity: AwareDatetime | None
    date_joined: AwareDatetime | None
