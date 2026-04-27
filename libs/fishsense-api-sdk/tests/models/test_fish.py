"""Tests for Fish model."""

from fishsense_api_sdk.models.fish import Fish


class TestFish:
    """Test suite for Fish model."""

    def test_fish_creation_with_all_fields(self):
        """Test creating a Fish instance with all fields."""
        fish = Fish(id=1, species_id=100)
        assert fish.id == 1
        assert fish.species_id == 100

    def test_fish_creation_with_none_values(self):
        """Test creating a Fish instance with None values."""
        fish = Fish(id=None, species_id=None)
        assert fish.id is None
        assert fish.species_id is None

    def test_fish_model_dump(self):
        """Test dumping Fish model to dictionary."""
        fish = Fish(id=1, species_id=100)
        data = fish.model_dump()
        assert isinstance(data, dict)
        assert data["id"] == 1
        assert data["species_id"] == 100

    def test_fish_model_validate(self):
        """Test validating Fish model from dictionary."""
        data = {"id": 2, "species_id": 200}
        fish = Fish.model_validate(data)
        assert fish.id == 2
        assert fish.species_id == 200

    def test_fish_optional_fields(self):
        """Test that Fish fields are optional."""
        fish = Fish(id=1, species_id=None)
        assert fish.id == 1
        assert fish.species_id is None

        fish2 = Fish(id=None, species_id=100)
        assert fish2.id is None
        assert fish2.species_id == 100
