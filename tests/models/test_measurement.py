"""Tests for Measurement model."""

import pytest

from fishsense_api_sdk.models.measurement import Measurement


class TestMeasurement:
    """Test suite for Measurement model."""

    def test_measurement_creation_with_all_fields(self):
        """Test creating a Measurement instance with all fields."""
        measurement = Measurement(id=1, length_m=0.5, image_id=10, fish_id=100)
        assert measurement.id == 1
        assert measurement.length_m == 0.5
        assert measurement.image_id == 10
        assert measurement.fish_id == 100

    def test_measurement_creation_with_none_values(self):
        """Test creating a Measurement instance with None values."""
        measurement = Measurement(
            id=None, length_m=None, image_id=None, fish_id=None
        )
        assert measurement.id is None
        assert measurement.length_m is None
        assert measurement.image_id is None
        assert measurement.fish_id is None

    def test_measurement_model_dump(self):
        """Test dumping Measurement model to dictionary."""
        measurement = Measurement(id=1, length_m=0.5, image_id=10, fish_id=100)
        data = measurement.model_dump()
        assert isinstance(data, dict)
        assert data["id"] == 1
        assert data["length_m"] == 0.5
        assert data["image_id"] == 10
        assert data["fish_id"] == 100

    def test_measurement_model_validate(self):
        """Test validating Measurement model from dictionary."""
        data = {"id": 2, "length_m": 1.2, "image_id": 20, "fish_id": 200}
        measurement = Measurement.model_validate(data)
        assert measurement.id == 2
        assert measurement.length_m == 1.2
        assert measurement.image_id == 20
        assert measurement.fish_id == 200

    def test_measurement_optional_fields(self):
        """Test that Measurement fields are optional."""
        measurement = Measurement(id=1, length_m=0.5, image_id=None, fish_id=None)
        assert measurement.id == 1
        assert measurement.length_m == 0.5
        assert measurement.image_id is None
        assert measurement.fish_id is None
