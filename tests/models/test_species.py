"""Tests for Species model."""

import pytest

from fishsense_api_sdk.models.species import Species


class TestSpecies:
    """Test suite for Species model."""

    def test_species_creation_with_all_fields(self):
        """Test creating a Species instance with all fields."""
        species = Species(
            id=1, scientific_name="Thunnus albacares", common_name="Yellowfin Tuna"
        )
        assert species.id == 1
        assert species.scientific_name == "Thunnus albacares"
        assert species.common_name == "Yellowfin Tuna"

    def test_species_creation_with_none_values(self):
        """Test creating a Species instance with None values."""
        species = Species(id=None, scientific_name=None, common_name=None)
        assert species.id is None
        assert species.scientific_name is None
        assert species.common_name is None

    def test_species_model_dump(self):
        """Test dumping Species model to dictionary."""
        species = Species(
            id=1, scientific_name="Thunnus albacares", common_name="Yellowfin Tuna"
        )
        data = species.model_dump()
        assert isinstance(data, dict)
        assert data["id"] == 1
        assert data["scientific_name"] == "Thunnus albacares"
        assert data["common_name"] == "Yellowfin Tuna"

    def test_species_model_validate(self):
        """Test validating Species model from dictionary."""
        data = {
            "id": 2,
            "scientific_name": "Seriola lalandi",
            "common_name": "Yellowtail Amberjack",
        }
        species = Species.model_validate(data)
        assert species.id == 2
        assert species.scientific_name == "Seriola lalandi"
        assert species.common_name == "Yellowtail Amberjack"

    def test_species_optional_fields(self):
        """Test that Species fields are optional."""
        species = Species(id=1, scientific_name="Thunnus albacares", common_name=None)
        assert species.id == 1
        assert species.scientific_name == "Thunnus albacares"
        assert species.common_name is None
