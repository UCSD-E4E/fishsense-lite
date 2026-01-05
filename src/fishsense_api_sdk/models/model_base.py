"""Base model class for all models in the Fishsense API SDK."""

from abc import ABC
from datetime import datetime

from pydantic import BaseModel, field_validator


class ModelBase(ABC, BaseModel):
    """Base model class with common functionality for all models."""

    @field_validator("*", mode="before")
    @classmethod
    def parse_date_fields(cls, v, info):
        """Validator to parse date fields from ISO format strings to datetime objects."""
        if info.field_name and "date" in info.field_name.lower():
            if isinstance(v, str):
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v
