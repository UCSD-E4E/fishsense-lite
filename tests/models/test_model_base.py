"""Tests for ModelBase class."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from fishsense_api_sdk.models.model_base import ModelBase


class TestModel(ModelBase):
    """Test model to validate ModelBase functionality."""

    id: int
    name: str
    created_date: datetime | None = None
    updated_date: datetime | None = None


class TestModelBase:
    """Test suite for ModelBase class."""

    def test_model_creation(self):
        """Test that a model can be created with basic fields."""
        model = TestModel(id=1, name="Test")
        assert model.id == 1
        assert model.name == "Test"

    def test_date_field_parsing_from_iso_string(self):
        """Test that date fields are parsed from ISO format strings."""
        model = TestModel(
            id=1,
            name="Test",
            created_date="2024-01-15T10:30:00Z",
            updated_date="2024-01-16T15:45:00+00:00",
        )
        assert isinstance(model.created_date, datetime)
        assert isinstance(model.updated_date, datetime)
        assert model.created_date.tzinfo is not None
        assert model.updated_date.tzinfo is not None

    def test_date_field_parsing_with_datetime_object(self):
        """Test that datetime objects are passed through without modification."""
        now = datetime.now(timezone.utc)
        model = TestModel(id=1, name="Test", created_date=now)
        assert model.created_date == now
        assert isinstance(model.created_date, datetime)

    def test_non_date_field_not_parsed(self):
        """Test that non-date fields are not affected by date parsing."""
        model = TestModel(id=1, name="Test with date in name")
        assert model.name == "Test with date in name"
        assert isinstance(model.name, str)

    def test_model_dump(self):
        """Test that model can be dumped to dictionary."""
        model = TestModel(id=1, name="Test", created_date="2024-01-15T10:30:00Z")
        data = model.model_dump()
        assert isinstance(data, dict)
        assert data["id"] == 1
        assert data["name"] == "Test"
        assert isinstance(data["created_date"], datetime)

    def test_model_validation(self):
        """Test that model validation works correctly."""
        with pytest.raises(ValidationError):
            TestModel(id="invalid", name="Test")
