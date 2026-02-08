"""Tests for UserClient class."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

from fishsense_api_sdk.clients.user_client import UserClient
from fishsense_api_sdk.models.user import User


class TestUserClient:
    """Test suite for UserClient class."""

    async def test_get_by_id(self):
        """Test getting a user by ID."""
        semaphore = asyncio.Semaphore(10)
        client = UserClient(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 1,
            "label_studio_id": 100,
            "email": "test@example.com",
            "first_name": "John",
            "last_name": "Doe",
            "last_activity": None,
            "date_joined": None,
        }
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                user = await client.get_by_id(1)
                assert isinstance(user, User)
                assert user.id == 1
                assert user.email == "test@example.com"
                mock_get.assert_called_once_with("/api/v1/users/1")

    async def test_get_by_id_not_found(self):
        """Test getting a user by ID when not found."""
        semaphore = asyncio.Semaphore(10)
        client = UserClient(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = None
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                user = await client.get_by_id(999)
                assert user is None

    async def test_get_by_email(self):
        """Test getting a user by email."""
        semaphore = asyncio.Semaphore(10)
        client = UserClient(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 1,
            "label_studio_id": 100,
            "email": "test@example.com",
            "first_name": "John",
            "last_name": "Doe",
            "last_activity": None,
            "date_joined": None,
        }
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                user = await client.get_by_email("test@example.com")
                assert isinstance(user, User)
                assert user.email == "test@example.com"
                mock_get.assert_called_once_with("/api/v1/users/email/test@example.com")

    async def test_get_by_email_not_found(self):
        """Test getting a user by email when not found."""
        semaphore = asyncio.Semaphore(10)
        client = UserClient(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = None
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                user = await client.get_by_email("notfound@example.com")
                assert user is None

    async def test_get_by_label_studio_id(self):
        """Test getting a user by Label Studio ID."""
        semaphore = asyncio.Semaphore(10)
        client = UserClient(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 1,
            "label_studio_id": 100,
            "email": "test@example.com",
            "first_name": "John",
            "last_name": "Doe",
            "last_activity": None,
            "date_joined": None,
        }
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                user = await client.get_by_label_studio_id(100)
                assert isinstance(user, User)
                assert user.label_studio_id == 100
                mock_get.assert_called_once_with("/api/v1/users/label-studio/100")

    async def test_list_all(self):
        """Test listing all users."""
        semaphore = asyncio.Semaphore(10)
        client = UserClient(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": 1,
                "label_studio_id": 100,
                "email": "user1@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "last_activity": None,
                "date_joined": None,
            },
            {
                "id": 2,
                "label_studio_id": 200,
                "email": "user2@example.com",
                "first_name": "Jane",
                "last_name": "Smith",
                "last_activity": None,
                "date_joined": None,
            },
        ]
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                users = await client.list_all()
                assert isinstance(users, list)
                assert len(users) == 2
                assert all(isinstance(u, User) for u in users)
                mock_get.assert_called_once_with("/api/v1/users/")

    async def test_list_all_empty(self):
        """Test listing all users when none exist."""
        semaphore = asyncio.Semaphore(10)
        client = UserClient(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = None
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                users = await client.list_all()
                assert users is None

    async def test_post_user(self):
        """Test creating a new user."""
        semaphore = asyncio.Semaphore(10)
        client = UserClient(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        user = User(
            id=None,
            label_studio_id=100,
            email="newuser@example.com",
            first_name="New",
            last_name="User",
            last_activity=None,
            date_joined=None,
        )
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = 1
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            async with client:
                user_id = await client.post(user)
                assert user_id == 1
                mock_post.assert_called_once()

    async def test_put_user(self):
        """Test updating a user."""
        semaphore = asyncio.Semaphore(10)
        client = UserClient(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        user = User(
            id=1,
            label_studio_id=100,
            email="updated@example.com",
            first_name="Updated",
            last_name="User",
            last_activity=None,
            date_joined=None,
        )
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = 1
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_put", new_callable=AsyncMock) as mock_put:
            mock_put.return_value = mock_response

            async with client:
                user_id = await client.put(user)
                assert user_id == 1
                mock_put.assert_called_once()
                call_args = mock_put.call_args
                assert call_args[0][0] == "/api/v1/users/1"
