"""Phase 6 regression guards for the SDK's httpx lifecycle.

The api-workflow-worker fans out hundreds of concurrent SDK calls per
sync activity. If `ClientBase` ever regressed to constructing a new
`httpx.AsyncClient` per call, every request would pay a fresh
TCP+TLS handshake — exactly the failure mode that caused the prod
sync timeouts (Phase 1).

These tests assert the contract:
  * Exactly one `httpx.AsyncClient` is constructed per `__aenter__`.
  * It's reused across all `_get/_post/_put` calls inside the context.
  * `aclose()` is called exactly once on `__aexit__`.
"""
# pylint: disable=protected-access

import asyncio
from unittest.mock import AsyncMock, Mock, patch

from fishsense_api_sdk.clients.client_base import ClientBase


class _ImplClient(ClientBase):  # pylint: disable=too-few-public-methods
    pass


async def test_one_httpx_client_constructed_per_context_entry():
    semaphore = asyncio.Semaphore(10)
    client = _ImplClient(
        base_url="http://test.com",
        username=None,
        password=None,
        timeout=10,
        semaphore=semaphore,
    )

    mock_response = Mock(status_code=200)

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_instance = AsyncMock()
        mock_instance.get = AsyncMock(return_value=mock_response)
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_instance.put = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_instance

        async with client:
            for _ in range(50):
                await client._get("/x")
            for _ in range(50):
                await client._post("/x", json={})
            for _ in range(50):
                await client._put("/x", json={})

        # Critical: 150 calls, but the AsyncClient class was instantiated
        # at most once.
        assert mock_client_class.call_count == 1
        # Closed exactly once on __aexit__.
        mock_instance.aclose.assert_awaited_once()


async def test_concurrent_calls_share_one_httpx_client():
    semaphore = asyncio.Semaphore(20)
    client = _ImplClient(
        base_url="http://test.com",
        username=None,
        password=None,
        timeout=10,
        semaphore=semaphore,
    )

    mock_response = Mock(status_code=200)

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_instance = AsyncMock()
        mock_instance.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_instance

        async with client:
            await asyncio.gather(*(client._get(f"/x/{i}") for i in range(40)))

        assert mock_client_class.call_count == 1
        assert mock_instance.get.await_count == 40
