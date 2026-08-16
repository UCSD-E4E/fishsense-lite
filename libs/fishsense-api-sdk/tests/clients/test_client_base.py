"""Tests for ClientBase class."""
# pylint: disable=protected-access

import asyncio
import base64
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from fishsense_api_sdk.clients import client_base
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
            mock_client_instance.request = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client_instance

            async with client:
                response = await client._get("/test")
                assert response == mock_response
                mock_client_instance.request.assert_called_once()
                call_kwargs = mock_client_instance.request.call_args[1]
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
            mock_client_instance.request = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client_instance

            async with client:
                response = await client._get("/test")
                assert response == mock_response
                mock_client_instance.request.assert_called_once()
                call_kwargs = mock_client_instance.request.call_args[1]
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
            mock_client_instance.request = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client_instance

            async with client:
                response = await client._post("/test", json={"data": "value"})
                assert response == mock_response
                mock_client_instance.request.assert_called_once()

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
            mock_client_instance.request = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client_instance

            async with client:
                response = await client._put("/test", json={"data": "updated"})
                assert response == mock_response
                mock_client_instance.request.assert_called_once()

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

    # ── retry contract ────────────────────────────────────────────────
    #
    # The old `@retry(exceptions=HTTPStatusError, tries=3)` decorators were
    # dead twice over: `retry` is synchronous, so decorating an `async def`
    # returned the coroutine before it could raise, and `HTTPStatusError` is
    # only produced by `raise_for_status()`, which the callers invoke — never
    # `_get`. Measured: one attempt, not three.
    #
    # That mattered beyond the SDK. `_retry_policies.SDK_FAIL_FAST_RETRY_POLICY`
    # marks `HTTPStatusError` NON-retryable, justified by "the SDK's
    # httpx-level @retry already absorbed any transient status". It absorbed
    # nothing, so a single transient 5xx failed the whole activity with no
    # retry at any layer.

    async def _client(self):
        return TestClientImpl(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=asyncio.Semaphore(10),
        )

    async def test_get_retries_transient_5xx_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(client_base, "_INITIAL_DELAY_S", 0)
        client = await self._client()
        responses = [Mock(status_code=503), Mock(status_code=200)]

        with patch("httpx.AsyncClient") as mock_client_class:
            instance = AsyncMock()
            instance.request = AsyncMock(side_effect=responses)
            mock_client_class.return_value = instance

            async with client:
                response = await client._get("/test")

            assert response.status_code == 200
            assert instance.request.call_count == 2

    async def test_get_gives_up_after_max_attempts_and_returns_the_response(
        self, monkeypatch
    ):
        """Exhausting retries returns the failing response rather than raising:
        callers inspect `status_code` (404-tolerance in `dive_client.get`) and
        call `raise_for_status()` themselves."""
        monkeypatch.setattr(client_base, "_INITIAL_DELAY_S", 0)
        client = await self._client()

        with patch("httpx.AsyncClient") as mock_client_class:
            instance = AsyncMock()
            instance.request = AsyncMock(return_value=Mock(status_code=503))
            mock_client_class.return_value = instance

            async with client:
                response = await client._get("/test")

            assert response.status_code == 503
            assert instance.request.call_count == client_base._MAX_ATTEMPTS

    async def test_get_does_not_retry_client_errors(self, monkeypatch):
        """A 404 is an answer, not a blip — retrying it just adds latency."""
        monkeypatch.setattr(client_base, "_INITIAL_DELAY_S", 0)
        client = await self._client()

        with patch("httpx.AsyncClient") as mock_client_class:
            instance = AsyncMock()
            instance.request = AsyncMock(return_value=Mock(status_code=404))
            mock_client_class.return_value = instance

            async with client:
                response = await client._get("/test")

            assert response.status_code == 404
            assert instance.request.call_count == 1

    async def test_get_retries_transport_errors(self, monkeypatch):
        monkeypatch.setattr(client_base, "_INITIAL_DELAY_S", 0)
        client = await self._client()

        with patch("httpx.AsyncClient") as mock_client_class:
            instance = AsyncMock()
            instance.request = AsyncMock(
                side_effect=[httpx.ReadTimeout("slow"), Mock(status_code=200)]
            )
            mock_client_class.return_value = instance

            async with client:
                response = await client._get("/test")

            assert response.status_code == 200
            assert instance.request.call_count == 2

    async def test_get_reraises_a_persistent_transport_error(self, monkeypatch):
        monkeypatch.setattr(client_base, "_INITIAL_DELAY_S", 0)
        client = await self._client()

        with patch("httpx.AsyncClient") as mock_client_class:
            instance = AsyncMock()
            instance.request = AsyncMock(side_effect=httpx.ConnectError("down"))
            mock_client_class.return_value = instance

            async with client:
                with pytest.raises(httpx.ConnectError):
                    await client._get("/test")

            assert instance.request.call_count == client_base._MAX_ATTEMPTS

    @pytest.mark.parametrize("verb", ["_put", "_delete"])
    async def test_idempotent_verbs_retry_5xx(self, monkeypatch, verb):
        monkeypatch.setattr(client_base, "_INITIAL_DELAY_S", 0)
        client = await self._client()

        with patch("httpx.AsyncClient") as mock_client_class:
            instance = AsyncMock()
            instance.request = AsyncMock(
                side_effect=[Mock(status_code=502), Mock(status_code=200)]
            )
            mock_client_class.return_value = instance

            async with client:
                response = await getattr(client, verb)("/test")

            assert response.status_code == 200
            assert instance.request.call_count == 2

    async def test_post_does_not_retry_5xx(self, monkeypatch):
        """POST is NOT idempotent here. `post_species` / `post_fish` create
        rows, so a 5xx that the server had already applied would be duplicated
        by a retry. The server may have committed before failing to respond —
        we cannot tell from the status alone, so we do not retry."""
        monkeypatch.setattr(client_base, "_INITIAL_DELAY_S", 0)
        client = await self._client()

        with patch("httpx.AsyncClient") as mock_client_class:
            instance = AsyncMock()
            instance.request = AsyncMock(return_value=Mock(status_code=503))
            mock_client_class.return_value = instance

            async with client:
                response = await client._post("/test", {"a": 1})

            assert response.status_code == 503
            assert instance.request.call_count == 1

    async def test_post_retries_only_connect_errors(self, monkeypatch):
        """A ConnectError proves the request never reached the server, so
        replaying it cannot duplicate a write. A ReadTimeout does not — the
        request may have been applied — so that one is left alone."""
        monkeypatch.setattr(client_base, "_INITIAL_DELAY_S", 0)
        client = await self._client()

        with patch("httpx.AsyncClient") as mock_client_class:
            instance = AsyncMock()
            instance.request = AsyncMock(
                side_effect=[httpx.ConnectError("refused"), Mock(status_code=201)]
            )
            mock_client_class.return_value = instance

            async with client:
                response = await client._post("/test", {"a": 1})

            assert response.status_code == 201
            assert instance.request.call_count == 2

    async def test_post_does_not_retry_read_timeouts(self, monkeypatch):
        monkeypatch.setattr(client_base, "_INITIAL_DELAY_S", 0)
        client = await self._client()

        with patch("httpx.AsyncClient") as mock_client_class:
            instance = AsyncMock()
            instance.request = AsyncMock(side_effect=httpx.ReadTimeout("slow"))
            mock_client_class.return_value = instance

            async with client:
                with pytest.raises(httpx.ReadTimeout):
                    await client._post("/test", {"a": 1})

            assert instance.request.call_count == 1

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
            mock_client_instance.request = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client_instance

            async with client:
                # Verify semaphore is available before request
                assert semaphore._value == 1
                # Make request
                await client._get("/test")
                # Verify semaphore is available after request completes
                assert semaphore._value == 1
