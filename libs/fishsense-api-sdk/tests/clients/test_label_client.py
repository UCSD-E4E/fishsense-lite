"""Tests for LabelClient class — focused on the 404→None contract for
single-resource getters. Mirrors the same regression the dive laser-extrinsics
endpoint exhibited.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

from fishsense_api_sdk.clients.label_client import LabelClient


def _make_client() -> LabelClient:
    return LabelClient(
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


class TestLabelClient:
    """Test suite for LabelClient class."""

    async def test_get_dive_slate_label_returns_none_on_404(self):
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert await client.get_dive_slate_label(999) is None

    async def test_get_headtail_label_by_image_id_returns_none_on_404(self):
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert await client.get_headtail_label(image_id=999) is None

    async def test_get_headtail_label_by_label_studio_id_returns_none_on_404(self):
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert (
                    await client.get_headtail_label(label_studio_id=999) is None
                )

    async def test_get_laser_label_by_image_id_returns_none_on_404(self):
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert await client.get_laser_label(image_id=999) is None

    async def test_get_laser_label_by_label_studio_id_returns_none_on_404(self):
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert await client.get_laser_label(label_studio_id=999) is None

    async def test_get_species_label_returns_none_on_404(self):
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert await client.get_species_label(999) is None
