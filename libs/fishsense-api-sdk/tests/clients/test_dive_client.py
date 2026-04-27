"""Tests for DiveClient class."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from fishsense_api_sdk.clients.dive_client import DiveClient
from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.laser_extrinsics import LaserExtrinsics
from fishsense_api_sdk.models.priority import Priority


def _make_client() -> DiveClient:
    return DiveClient(
        base_url="http://test.com",
        username="testuser",
        password="testpass",
        timeout=10,
        semaphore=asyncio.Semaphore(10),
    )


def _dive_payload(dive_id: int = 1, name: str = "test-dive") -> dict:
    return {
        "id": dive_id,
        "name": name,
        "path": "/data/dives/test",
        "dive_datetime": "2026-01-01T00:00:00Z",
        "priority": "HIGH",
        "flip_dive_slate": False,
        "camera_id": 12,
        "dive_slate_id": None,
    }


class TestDiveClient:
    """Test suite for DiveClient class."""

    async def test_get_single_dive_returns_none_on_404(self):
        """A 404 from GET /dives/{id} returns None instead of raising."""
        client = _make_client()
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status = Mock(
            side_effect=AssertionError(
                "raise_for_status must not be called for the 404 case"
            )
        )

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                result = await client.get(dive_id=999)
                assert result is None

    async def test_get_single_dive_returns_none_on_null_body(self):
        """Backwards-compat: a 200 with null body returns None."""
        client = _make_client()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = None
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                result = await client.get(dive_id=999)
                assert result is None

    async def test_get_single_dive_success(self):
        client = _make_client()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = _dive_payload(dive_id=42)
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                result = await client.get(dive_id=42)
                assert isinstance(result, Dive)
                assert result.id == 42
                assert result.priority == Priority.HIGH
                mock_get.assert_called_once_with("/api/v1/dives/42")

    async def test_get_all_dives_success(self):
        client = _make_client()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            _dive_payload(dive_id=1, name="d1"),
            _dive_payload(dive_id=2, name="d2"),
        ]
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                result = await client.get()
                assert isinstance(result, list)
                assert len(result) == 2
                assert all(isinstance(d, Dive) for d in result)
                assert [d.id for d in result] == [1, 2]
                mock_get.assert_called_once_with("/api/v1/dives/")

    async def test_get_all_dives_empty(self):
        client = _make_client()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = None
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                result = await client.get()
                assert result is None

    async def test_get_canonical_success(self):
        client = _make_client()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [_dive_payload(dive_id=7)]
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                result = await client.get_canonical()
                assert isinstance(result, list)
                assert len(result) == 1
                assert result[0].id == 7
                mock_get.assert_called_once_with("/api/v1/canonical/dives/")

    async def test_get_canonical_empty(self):
        client = _make_client()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = None
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                result = await client.get_canonical()
                assert result is None

    async def test_get_laser_extrinsics_returns_none_on_404(self):
        """A 404 returns None and never calls raise_for_status — this is the
        original reported bug. Callers rely on `if not extrinsics:` working.
        """
        client = _make_client()
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status = Mock(
            side_effect=AssertionError(
                "raise_for_status must not be called for the 404 case"
            )
        )

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                result = await client.get_laser_extrinsics(439)
                assert result is None
                mock_get.assert_called_once_with(
                    "/api/v1/dives/439/laser-extrinsics/"
                )

    async def test_get_laser_extrinsics_returns_none_on_null_body(self):
        client = _make_client()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = None
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                result = await client.get_laser_extrinsics(439)
                assert result is None

    async def test_get_laser_extrinsics_success(self):
        client = _make_client()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 7,
            "laser_position": [0.1, 0.2, 0.3],
            "laser_axis": [0.0, 0.0, 1.0],
            "created_at": None,
            "dive_id": 439,
            "camera_id": 12,
        }
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                result = await client.get_laser_extrinsics(439)
                assert isinstance(result, LaserExtrinsics)
                assert result.id == 7
                assert result.dive_id == 439
                assert result.camera_id == 12

    async def test_put_laser_extrinsics_round_trip(self):
        """put_laser_extrinsics must serialize through _to_internal() and
        return the dive id from the response body.
        """
        client = _make_client()
        extrinsics = LaserExtrinsics(
            laser_position=np.array([1.0, 2.0, 3.0]),
            laser_axis=np.array([0.0, 0.0, 1.0]),
            dive_id=439,
            camera_id=12,
            id=None,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = 439
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_put", new_callable=AsyncMock) as mock_put:
            mock_put.return_value = mock_response

            async with client:
                returned_id = await client.put_laser_extrinsics(439, extrinsics)
                assert returned_id == 439
                mock_put.assert_called_once()
                endpoint, *_ = mock_put.call_args.args
                assert endpoint == "/api/v1/dives/439/laser-extrinsics/"
                payload = mock_put.call_args.kwargs["json"]
                assert payload["dive_id"] == 439
                assert payload["camera_id"] == 12
                assert payload["laser_position"] == [1.0, 2.0, 3.0]
                assert payload["laser_axis"] == [0.0, 0.0, 1.0]
