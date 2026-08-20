"""Tests for ImageClient class."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from fishsense_api_sdk.clients.image_client import ImageClient
from fishsense_api_sdk.models.image import Image


def _make_client() -> ImageClient:
    return ImageClient(
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


class TestImageClient:
    """Test suite for ImageClient class."""

    async def test_get_by_image_id_returns_none_on_404(self):
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert await client.get(image_id=999) is None

    async def test_get_by_checksum_returns_none_on_404(self):
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert await client.get(checksum="deadbeef") is None


class TestImageClientPost:
    """`images.post` and `images.lookup_checksums` — the image half of ingest."""

    async def test_post_sends_the_image_to_the_dive_scoped_route(self):
        client = _make_client()
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value=7)

        image = Image.model_validate(
            {
                "id": None,
                "path": "d/7/P8210001.ORF",
                "taken_datetime": "2024-08-21T08:56:51Z",
                "checksum": "45dc5a454b35601b9dafabf24822195d",
                "is_canonical": False,
                "dive_id": None,
                "camera_id": 3,
            }
        )

        with patch.object(client, "_post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            image_id = await client.post(7, image)

        assert image_id == 7
        assert mock_post.call_args.args[0] == "/api/v1/dives/7/images/"

    async def test_post_strips_is_canonical_so_the_server_computes_it(self):
        """`is_canonical` is computed server-side (first-checksum-wins), and an
        explicit value in the body overrides that computation.

        `Image.is_canonical` is a **required** field on the SDK model, so every
        instance carries a value and `exclude_unset` cannot keep it off the
        wire. Posting it unconditionally would mark every duplicate frame
        canonical and silently break the dives-64/66 distinction — so the
        client strips it unless asked not to.
        """
        client = _make_client()
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value=1)

        image = Image(
            id=None,
            path="d/7/P8210001.ORF",
            taken_datetime=datetime(2024, 8, 21, 8, 56, 51, tzinfo=timezone.utc),
            checksum="45dc5a454b35601b9dafabf24822195d",
            is_canonical=True,  # a plausible-looking default that must NOT ship
            dive_id=None,
            camera_id=3,
        )

        with patch.object(client, "_post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.post(7, image)

        assert "is_canonical" not in mock_post.call_args.kwargs["json"]

    async def test_post_sends_is_canonical_when_the_caller_opts_in(self):
        """The operator override — deliberate, and it has to travel."""
        client = _make_client()
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value=1)

        image = Image(
            id=None,
            path="d/66/P1.ORF",
            taken_datetime=datetime(2024, 8, 21, 8, 56, 51, tzinfo=timezone.utc),
            checksum="45dc5a454b35601b9dafabf24822195d",
            is_canonical=True,
            dive_id=None,
            camera_id=3,
        )

        with patch.object(client, "_post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.post(66, image, set_canonical=True)

        assert mock_post.call_args.kwargs["json"]["is_canonical"] is True

    async def test_lookup_checksums_posts_the_batch_and_returns_the_mapping(self):
        client = _make_client()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(
            return_value={
                "a" * 32: [{"image_id": 1, "dive_id": 64, "is_canonical": True}],
                "b" * 32: [],
            }
        )

        with patch.object(client, "_post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.lookup_checksums(["a" * 32, "b" * 32])

        assert mock_post.call_args.args[0] == "/api/v1/images/checksums/lookup"
        assert mock_post.call_args.kwargs["json"] == ["a" * 32, "b" * 32]
        assert result["b" * 32] == []
        assert result["a" * 32][0]["dive_id"] == 64


class TestImageClientLaserDepth:
    """`images.get_laser_depth` / `put_laser_depth` / `get_laser_depths`.

    The distance to an image's laser dot. Written by the data-worker (the
    projection kernel lives in `fishsense-core`, which only that service
    depends on) and read by anything that wants range without re-deriving it.
    """

    async def test_get_laser_depth_returns_none_on_404(self):
        """No depth yet is a normal pipeline state for an image, so callers
        get None rather than an exception to catch."""
        client = _make_client()
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_404()
            async with client:
                assert await client.get_laser_depth(11) is None

    async def test_get_laser_depth_parses_the_row(self):
        from fishsense_api_sdk.models.laser_depth import (  # pylint: disable=import-outside-toplevel
            LaserDepth,
        )

        client = _make_client()
        response = Mock()
        response.status_code = 200
        response.raise_for_status = Mock()
        response.json = Mock(
            return_value={
                "id": 3,
                "depth_m": 2.14,
                "range_m": 2.19,
                "image_id": 11,
                "laser_label_id": 5,
                "laser_extrinsics_id": 7,
            }
        )
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = response
            async with client:
                depth = await client.get_laser_depth(11)

        assert mock_get.call_args.args[0] == "/api/v1/images/11/laser-depth/"
        assert isinstance(depth, LaserDepth)
        assert (depth.depth_m, depth.range_m) == (2.14, 2.19)
        assert (depth.laser_label_id, depth.laser_extrinsics_id) == (5, 7)

    async def test_put_laser_depth_targets_the_image_route(self):
        from fishsense_api_sdk.models.laser_depth import (  # pylint: disable=import-outside-toplevel
            LaserDepth,
        )

        client = _make_client()
        response = Mock()
        response.status_code = 201
        response.raise_for_status = Mock()
        response.json = Mock(return_value=3)
        with patch.object(client, "_put", new_callable=AsyncMock) as mock_put:
            mock_put.return_value = response
            async with client:
                depth_id = await client.put_laser_depth(
                    11,
                    LaserDepth(
                        depth_m=2.14,
                        range_m=2.19,
                        laser_label_id=5,
                        laser_extrinsics_id=7,
                    ),
                )

        assert depth_id == 3
        assert mock_put.call_args.args[0] == "/api/v1/images/11/laser-depth/"
        assert mock_put.call_args.kwargs["json"]["depth_m"] == 2.14

    async def test_get_laser_depths_returns_empty_list_when_dive_has_none(self):
        """Dive-scoped read, unlike the per-image one, treats "none" as an
        empty collection — the compute activity iterates it to see what it has
        already done, and an absent dive is not an error there."""
        client = _make_client()
        response = Mock()
        response.status_code = 200
        response.raise_for_status = Mock()
        response.json = Mock(return_value=[])
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = response
            async with client:
                assert await client.get_laser_depths(7) == []

        assert mock_get.call_args.args[0] == "/api/v1/dives/7/laser-depths/"
