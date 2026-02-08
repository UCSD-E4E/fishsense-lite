"""Tests for ClientBase class."""
# pylint: disable=protected-access

import asyncio
import base64
from unittest.mock import AsyncMock, Mock, patch

import pytest

from fishsense_api_sdk.clients.client_base import ClientBase


class TestClientImpl(ClientBase):  # pylint: disable=too-few-public-methods
    """Test implementation of ClientBase for testing."""


class TestClientBase:
    """Test suite for ClientBase class."""

    def test_client_initialization(self):
        """Test that ClientBase can be initialized."""
        semaphore = asyncio.Semaphore(10)
        client = TestClientImpl(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )
        assert client.base_url == "http://test.com"
        assert client.timeout == 10
        assert client.semaphore == semaphore

    def test_client_initialization_without_credentials(self):
        """Test that ClientBase can be initialized without credentials."""
        semaphore = asyncio.Semaphore(10)
        client = TestClientImpl(
            base_url="http://test.com",
            username=None,
            password=None,
            timeout=10,
            semaphore=semaphore,
        )
        assert client.base_url == "http://test.com"

    async def test_context_manager_entry(self):
        """Test entering async context manager."""
        semaphore = asyncio.Semaphore(10)
        client = TestClientImpl(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        async with client as c:
            assert c == client

    async def test_context_manager_exit(self):
        """Test exiting async context manager."""
        semaphore = asyncio.Semaphore(10)
        client = TestClientImpl(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        async with client:
            pass
        # Should not raise any exceptions

    async def test_get_request_with_authentication(self):
        """Test GET request with authentication headers."""
        semaphore = asyncio.Semaphore(10)
        client = TestClientImpl(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client_instance

            async with client:
                response = await client._get("/test")
                assert response == mock_response
                mock_client_instance.get.assert_called_once()
                call_kwargs = mock_client_instance.get.call_args[1]
                assert "Authorization" in call_kwargs["headers"]
                auth_header = call_kwargs["headers"]["Authorization"]
                assert auth_header.startswith("Basic ")
                # Verify the encoded credentials
                encoded_creds = auth_header.replace("Basic ", "")
                decoded_creds = base64.b64decode(encoded_creds).decode("utf-8")
                assert decoded_creds == "testuser:testpass"

    async def test_get_request_without_authentication(self):
        """Test GET request without authentication headers."""
        semaphore = asyncio.Semaphore(10)
        client = TestClientImpl(
            base_url="http://test.com",
            username=None,
            password=None,
            timeout=10,
            semaphore=semaphore,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client_instance

            async with client:
                response = await client._get("/test")
                assert response == mock_response
                mock_client_instance.get.assert_called_once()
                call_kwargs = mock_client_instance.get.call_args[1]
                assert "Authorization" not in call_kwargs["headers"]

    async def test_post_request(self):
        """Test POST request."""
        semaphore = asyncio.Semaphore(10)
        client = TestClientImpl(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 1}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client_instance

            async with client:
                response = await client._post("/test", json={"data": "value"})
                assert response == mock_response
                mock_client_instance.post.assert_called_once()

    async def test_put_request(self):
        """Test PUT request."""
        semaphore = asyncio.Semaphore(10)
        client = TestClientImpl(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"updated": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.put = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client_instance

            async with client:
                response = await client._put("/test", json={"data": "updated"})
                assert response == mock_response
                mock_client_instance.put.assert_called_once()

    async def test_request_outside_context_raises_error(self):
        """Test that requests outside context manager raise RuntimeError."""
        semaphore = asyncio.Semaphore(10)
        client = TestClientImpl(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        with pytest.raises(RuntimeError, match="Client must be used within"):
            await client._get("/test")

        with pytest.raises(RuntimeError, match="Client must be used within"):
            await client._post("/test", json={"data": "value"})

        with pytest.raises(RuntimeError, match="Client must be used within"):
            await client._put("/test", json={"data": "value"})

    async def test_semaphore_is_used(self):
        """Test that semaphore is acquired during requests."""
        semaphore = asyncio.Semaphore(1)
        client = TestClientImpl(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        mock_response = Mock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client_instance

            async with client:
                # Verify semaphore is available before request
                assert semaphore._value == 1
                # Make request
                await client._get("/test")
                # Verify semaphore is available after request completes
                assert semaphore._value == 1
