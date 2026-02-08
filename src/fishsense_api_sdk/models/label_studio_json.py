"""Module defining the LabelStudioJson model for Fishsense API SDK."""

from pydantic import BaseModel, ConfigDict


class LabelStudioJson(BaseModel):
    """Model representing the JSON data from Label Studio."""

    model_config = ConfigDict(extra="allow")
