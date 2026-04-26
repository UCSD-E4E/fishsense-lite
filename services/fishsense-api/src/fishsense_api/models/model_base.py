"""Base model class for all models in the FishSense API."""

from abc import ABC
from datetime import datetime
from typing import get_args

from pydantic import model_validator
from sqlmodel import SQLModel


class ModelBase(ABC, SQLModel):
    """Base model class with common functionality for all models."""

    def __init__(self, **data):
        # Coerce ISO datetime strings for annotated datetime fields before pydantic parsing
        def _is_datetime_annotation(ann) -> bool:
            if ann is None:
                return False
            if ann is datetime:
                return True
            args = get_args(ann)
            if args:
                return any(_is_datetime_annotation(a) for a in args)
            return False

        ann = getattr(self.__class__, "__annotations__", {})
        for key, val in list(data.items()):
            field_info = ann.get(key)
            if field_info is None and hasattr(self.__class__, "model_fields"):
                mf = self.__class__.model_fields.get(key)
                if mf is not None:
                    field_info = getattr(mf, "annotation", None)
            if _is_datetime_annotation(field_info) or (
                isinstance(key, str)
                and ("date" in key.lower() or key.lower().endswith("_at"))
            ):
                if isinstance(val, str) and val:
                    try:
                        data[key] = datetime.fromisoformat(val.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        pass
                elif val == "":
                    data[key] = None

        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def parse_date_fields(cls, values: dict):
        """Model-level validator to coerce ISO datetime strings to datetimes.

        Runs before field parsing so it can accept raw input values (dicts, strs).
        """
        if not isinstance(values, dict):
            return values

        def _is_datetime_annotation(ann) -> bool:
            if ann is None:
                return False
            if ann is datetime:
                return True
            args = get_args(ann)
            if args:
                return any(_is_datetime_annotation(a) for a in args)
            return False

        for key, val in list(values.items()):
            # Determine declared annotation for this field
            field_info = getattr(cls, "__annotations__", {}).get(key)
            # Fallback to model_fields metadata when available
            if field_info is None and hasattr(cls, "model_fields"):
                mf = cls.model_fields.get(key)
                if mf is not None:
                    field_info = getattr(mf, "annotation", None)

            if _is_datetime_annotation(field_info) or (
                isinstance(key, str)
                and ("date" in key.lower() or key.lower().endswith("_at"))
            ):
                if isinstance(val, str) and val:
                    try:
                        values[key] = datetime.fromisoformat(val.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        # leave as-is; pydantic will raise if invalid
                        pass
                elif val == "":
                    values[key] = None

        return values
