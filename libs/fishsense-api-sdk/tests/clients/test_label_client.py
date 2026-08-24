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


class TestLabelClient:  # pylint: disable=too-many-public-methods
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

    async def test_put_laser_prediction_puts_to_image_endpoint(self):
        from fishsense_api_sdk.models.laser_prediction import (
            LaserPrediction,
        )

        client = _make_client()
        resp = Mock()
        resp.status_code = 201
        resp.json.return_value = 7
        resp.raise_for_status = Mock()
        with patch.object(client, "_put", new_callable=AsyncMock) as mock_put:
            mock_put.return_value = resp
            async with client:
                pid = await client.put_laser_prediction(
                    11, LaserPrediction(x=1.0, y=2.0, confidence=0.9)
                )
        assert pid == 7
        endpoint, *_ = mock_put.call_args.args
        assert endpoint == "/api/v1/images/11/laser-prediction/"

    async def test_get_laser_predictions_returns_list(self):
        client = _make_client()
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = [
            {"id": 1, "x": 1.0, "y": 2.0, "confidence": 0.9, "image_id": 11},
            {"id": 2, "x": None, "y": None, "confidence": 0.1, "image_id": 12},
        ]
        resp.raise_for_status = Mock()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = resp
            async with client:
                preds = await client.get_laser_predictions(1)
        assert [p.image_id for p in preds] == [11, 12]
        mock_get.assert_awaited_once_with("/api/v1/dives/1/laser-predictions/")

    async def test_get_laser_predictions_empty_on_null_body(self):
        client = _make_client()
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = None
        resp.raise_for_status = Mock()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = resp
            async with client:
                assert await client.get_laser_predictions(1) == []

    async def test_get_species_label_returns_none_on_404(self):
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert await client.get_species_label(999) is None

    async def test_get_species_label_by_label_studio_id_returns_none_on_404(self):
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert (
                    await client.get_species_label(label_studio_id=999) is None
                )

    # 404→None for the per-dive *plural* getters. The API returns 404 with
    # `Labels not found` when a dive has zero rows of a given label kind
    # (label_controller.py — `get_*_labels_for_dive`). That is a valid
    # steady state — e.g. a dive with valid laser labels but no
    # head/tail rows yet — and must not blow up the workflow that
    # called the SDK. Regression caught when the stage 5.1 head/tail
    # cascade fired against dive 58 (no head/tail rows yet) and
    # `resolve_headtail_preprocess_inputs_activity` 500'd.

    async def test_get_dive_slate_labels_returns_none_on_404(self):
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert await client.get_dive_slate_labels(999) is None

    async def test_get_headtail_labels_returns_none_on_404(self):
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert await client.get_headtail_labels(999) is None

    async def test_get_laser_labels_returns_none_on_404(self):
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert await client.get_laser_labels(999) is None

    async def test_get_species_labels_returns_none_on_404(self):
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert await client.get_species_labels(999) is None

    async def test_get_laser_label_studio_project_ids_hits_collection_endpoint(self):
        client = _make_client()
        response = Mock()
        response.status_code = 200
        response.raise_for_status = Mock()
        response.json = Mock(return_value=[101, 202, 303])
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = response
            async with client:
                ids = await client.get_laser_label_studio_project_ids()
        mock_get.assert_awaited_once_with(
            "/api/v1/labels/laser/label-studio-project-ids"
        )
        assert ids == [101, 202, 303]

    async def test_get_laser_label_studio_project_ids_handles_null_body(self):
        client = _make_client()
        response = Mock()
        response.status_code = 200
        response.raise_for_status = Mock()
        response.json = Mock(return_value=None)
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = response
            async with client:
                assert await client.get_laser_label_studio_project_ids() == []

    async def test_get_laser_label_studio_project_ids_passes_incomplete_flag(self):
        client = _make_client()
        response = Mock()
        response.status_code = 200
        response.raise_for_status = Mock()
        response.json = Mock(return_value=[5])
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = response
            async with client:
                await client.get_laser_label_studio_project_ids(incomplete=True)
        mock_get.assert_awaited_once_with(
            "/api/v1/labels/laser/label-studio-project-ids?incomplete=true"
        )

    async def test_get_headtail_label_studio_project_ids_hits_collection_endpoint(self):
        client = _make_client()
        response = Mock()
        response.status_code = 200
        response.raise_for_status = Mock()
        response.json = Mock(return_value=[11, 22])
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = response
            async with client:
                ids = await client.get_headtail_label_studio_project_ids()
        mock_get.assert_awaited_once_with(
            "/api/v1/labels/headtail/label-studio-project-ids"
        )
        assert ids == [11, 22]

    async def test_get_headtail_label_studio_project_ids_handles_null_body(self):
        client = _make_client()
        response = Mock()
        response.status_code = 200
        response.raise_for_status = Mock()
        response.json = Mock(return_value=None)
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = response
            async with client:
                assert await client.get_headtail_label_studio_project_ids() == []

    async def test_get_headtail_label_studio_project_ids_passes_incomplete_flag(self):
        client = _make_client()
        response = Mock()
        response.status_code = 200
        response.raise_for_status = Mock()
        response.json = Mock(return_value=[])
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = response
            async with client:
                await client.get_headtail_label_studio_project_ids(incomplete=True)
        mock_get.assert_awaited_once_with(
            "/api/v1/labels/headtail/label-studio-project-ids?incomplete=true"
        )

    async def test_get_species_label_studio_project_ids_hits_collection_endpoint(self):
        client = _make_client()
        response = Mock()
        response.status_code = 200
        response.raise_for_status = Mock()
        response.json = Mock(return_value=[31, 32])
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = response
            async with client:
                ids = await client.get_species_label_studio_project_ids()
        mock_get.assert_awaited_once_with(
            "/api/v1/labels/species/label-studio-project-ids"
        )
        assert ids == [31, 32]

    async def test_get_species_label_studio_project_ids_passes_incomplete_flag(self):
        client = _make_client()
        response = Mock()
        response.status_code = 200
        response.raise_for_status = Mock()
        response.json = Mock(return_value=[])
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = response
            async with client:
                await client.get_species_label_studio_project_ids(incomplete=True)
        mock_get.assert_awaited_once_with(
            "/api/v1/labels/species/label-studio-project-ids?incomplete=true"
        )

    async def test_get_dive_slate_label_studio_project_ids_hits_collection_endpoint(
        self,
    ):
        client = _make_client()
        response = Mock()
        response.status_code = 200
        response.raise_for_status = Mock()
        response.json = Mock(return_value=[41])
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = response
            async with client:
                ids = await client.get_dive_slate_label_studio_project_ids()
        mock_get.assert_awaited_once_with(
            "/api/v1/labels/dive-slate/label-studio-project-ids"
        )
        assert ids == [41]

    async def test_get_dive_slate_label_studio_project_ids_passes_incomplete_flag(
        self,
    ):
        client = _make_client()
        response = Mock()
        response.status_code = 200
        response.raise_for_status = Mock()
        response.json = Mock(return_value=[])
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = response
            async with client:
                await client.get_dive_slate_label_studio_project_ids(incomplete=True)
        mock_get.assert_awaited_once_with(
            "/api/v1/labels/dive-slate/label-studio-project-ids?incomplete=true"
        )
