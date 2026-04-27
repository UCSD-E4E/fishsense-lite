"""Tests for CameraClient class."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

from fishsense_api_sdk.clients.camera_client import CameraClient


def _make_client() -> CameraClient:
    return CameraClient(
        base_url="http://test.com",
        username="testuser",
        password="testpass",
        timeout=10,
        semaphore=asyncio.Semaphore(10),
    )


def _mock_404() -> Mock:
    response = Mock()
    response.status_code = 404
    response.raise_for_status = Mock(
        side_effect=AssertionError(
            "raise_for_status must not be called for the 404 case"
        )
    )
    return response


class TestCameraClient:
    """Test suite for CameraClient class."""

    async def test_get_single_camera_returns_none_on_404(self):
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert await client.get(camera_id=999) is None

    async def test_get_intrinsics_returns_none_on_404(self):
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert await client.get_intrinsics(999) is None
