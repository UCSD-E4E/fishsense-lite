"""Tests for User model."""

from datetime import datetime, timezone

from fishsense_api_sdk.models.user import User


class TestUser:
    """Test suite for User model."""

    def test_user_creation_with_all_fields(self):
        """Test creating a User instance with all fields."""
        now = datetime.now(timezone.utc)
        user = User(
            id=1,
            label_studio_id=100,
            email="test@example.com",
            first_name="John",
            last_name="Doe",
            last_activity=now,
            date_joined=now,
        )
        assert user.id == 1
        assert user.label_studio_id == 100
        assert user.email == "test@example.com"
        assert user.first_name == "John"
        assert user.last_name == "Doe"
        assert user.last_activity == now
        assert user.date_joined == now

    def test_user_creation_with_none_values(self):
        """Test creating a User instance with None values."""
        user = User(
            id=None,
            label_studio_id=None,
            email=None,
            first_name=None,
            last_name=None,
            last_activity=None,
            date_joined=None,
        )
        assert user.id is None
        assert user.label_studio_id is None
        assert user.email is None

    def test_user_model_dump(self):
        """Test dumping User model to dictionary."""
        user = User(
            id=1,
            label_studio_id=100,
            email="test@example.com",
            first_name="John",
            last_name="Doe",
            last_activity=None,
            date_joined=None,
        )
        data = user.model_dump()
        assert isinstance(data, dict)
        assert data["id"] == 1
        assert data["email"] == "test@example.com"

    def test_user_model_validate(self):
        """Test validating User model from dictionary."""
        data = {
            "id": 2,
            "label_studio_id": 200,
            "email": "test2@example.com",
            "first_name": "Jane",
            "last_name": "Smith",
            "last_activity": None,
            "date_joined": None,
        }
        user = User.model_validate(data)
        assert user.id == 2
        assert user.email == "test2@example.com"

    def test_user_date_field_parsing(self):
        """Test that date fields are parsed from ISO format strings."""
        data = {
            "id": 1,
            "label_studio_id": 100,
            "email": "test@example.com",
            "first_name": "John",
            "last_name": "Doe",
            "last_activity": "2024-01-15T10:30:00Z",
            "date_joined": "2024-01-01T08:00:00+00:00",
        }
        user = User.model_validate(data)
        assert isinstance(user.last_activity, datetime)
        assert isinstance(user.date_joined, datetime)
        assert user.last_activity.tzinfo is not None
        assert user.date_joined.tzinfo is not None
