"""Base model class for all models in the Fishsense API SDK."""

from abc import ABC
from datetime import datetime
from typing import Union, get_args, get_origin

from pydantic import BaseModel, field_validator


class ModelBase(ABC, BaseModel):
    """Base model class with common functionality for all models."""

    @field_validator("*", mode="before")
    @classmethod
    def parse_date_fields(cls, v, info):
        """Validator to parse date fields from ISO format strings to datetime objects."""
        # Prefer using the declared annotation/type for the field when available
        field_type = (
            getattr(info, "annotation", None)
            or getattr(info, "outer_type_", None)
            or getattr(info, "type_", None)
        )

        def _is_datetime_annotation(ann) -> bool:
            if ann is None:
                return False
            if ann is datetime:
                return True
            origin = get_origin(ann)
            if origin is Union:
                return any(_is_datetime_annotation(a) for a in get_args(ann))
            return False

        if _is_datetime_annotation(field_type) or (
            info.field_name
            and (
                "date" in info.field_name.lower()
                or info.field_name.lower().endswith("_at")
            )
        ):
            if isinstance(v, str) and v:
                # Handle trailing Z (UTC) by replacing with +00:00 for fromisoformat
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            # Allow empty string to be treated as None
            if v == "":
                return None
        return v
