"""Model for Label Studio Project."""

from pydantic import BaseModel


class LabelStudioProject(BaseModel):
    """Model for Label Studio Project."""

    id: str
    name: str
