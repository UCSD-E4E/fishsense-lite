"""Tests for main Client class."""

import asyncio

import pytest

from fishsense_api_sdk.client import Client
from fishsense_api_sdk.clients.camera_client import CameraClient
from fishsense_api_sdk.clients.dive_client import DiveClient
from fishsense_api_sdk.clients.dive_slate_client import DiveSlateClient
from fishsense_api_sdk.clients.fish_client import FishClient
from fishsense_api_sdk.clients.image_client import ImageClient
from fishsense_api_sdk.clients.label_client import LabelClient
from fishsense_api_sdk.clients.user_client import UserClient


class TestClient:
    """Test suite for main Client class."""

    def test_client_initialization(self):
        """Test that Client can be initialized."""
        client = Client(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            max_concurrent_requests=5,
        )
        assert client.base_url == "http://test.com"

    def test_client_properties(self):
        """Test that Client properties return correct client instances."""
        client = Client(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
        )

        assert isinstance(client.cameras, CameraClient)
        assert isinstance(client.dives, DiveClient)
        assert isinstance(client.dive_slates, DiveSlateClient)
        assert isinstance(client.fish, FishClient)
        assert isinstance(client.images, ImageClient)
        assert isinstance(client.labels, LabelClient)
        assert isinstance(client.users, UserClient)

    def test_client_default_timeout(self):
        """Test that Client uses default timeout."""
        client = Client(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
        )
        # Check that cameras client has the timeout
        assert client.cameras.timeout == 10

    def test_client_custom_timeout(self):
        """Test that Client uses custom timeout."""
        client = Client(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=30,
        )
        # Check that cameras client has the timeout
        assert client.cameras.timeout == 30

    def test_client_max_concurrent_requests(self):
        """Test that Client uses max_concurrent_requests."""
        client = Client(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            max_concurrent_requests=5,
        )
        # Check that the semaphore is shared
        assert client.cameras.semaphore._value == 5

    async def test_client_context_manager(self):
        """Test that Client can be used as async context manager."""
        client = Client(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
        )

        async with client as c:
            assert c == client

    async def test_client_context_manager_initializes_all_clients(self):
        """Test that entering context manager initializes all sub-clients."""
        client = Client(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
        )

        async with client:
            # All clients should be accessible within context
            assert client.cameras is not None
            assert client.dives is not None
            assert client.dive_slates is not None
            assert client.fish is not None
            assert client.images is not None
            assert client.labels is not None
            assert client.users is not None

    async def test_client_shared_semaphore(self):
        """Test that all sub-clients share the same semaphore."""
        client = Client(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            max_concurrent_requests=3,
        )

        # All clients should have the same semaphore instance
        assert client.cameras.semaphore is client.dives.semaphore
        assert client.dives.semaphore is client.fish.semaphore
        assert client.fish.semaphore is client.images.semaphore
        assert client.images.semaphore is client.labels.semaphore
        assert client.labels.semaphore is client.users.semaphore
