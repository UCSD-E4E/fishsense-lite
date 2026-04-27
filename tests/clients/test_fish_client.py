"""Tests for FishClient class."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

from fishsense_api_sdk.clients.fish_client import FishClient
from fishsense_api_sdk.models.fish import Fish
from fishsense_api_sdk.models.measurement import Measurement
from fishsense_api_sdk.models.species import Species


class TestFishClient:
    """Test suite for FishClient class."""

    async def test_get_single_fish(self):
        """Test getting a single fish by ID."""
        semaphore = asyncio.Semaphore(10)
        client = FishClient(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "species_id": 100}
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                fish = await client.get(fish_id=1)
                assert isinstance(fish, Fish)
                assert fish.id == 1
                assert fish.species_id == 100
                mock_get.assert_called_once_with("/api/v1/fish/1")

    async def test_get_single_fish_not_found(self):
        """Test getting a single fish that doesn't exist."""
        semaphore = asyncio.Semaphore(10)
        client = FishClient(
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
                fish = await client.get(fish_id=999)
                assert fish is None

    async def test_get_single_fish_returns_none_on_404(self):
        """A 404 from GET /fish/{id} returns None instead of raising."""
        semaphore = asyncio.Semaphore(10)
        client = FishClient(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

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
                fish = await client.get(fish_id=999)
                assert fish is None

    async def test_get_all_fish(self):
        """Test getting all fish."""
        semaphore = asyncio.Semaphore(10)
        client = FishClient(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": 1, "species_id": 100},
            {"id": 2, "species_id": 200},
        ]
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                fish_list = await client.get()
                assert isinstance(fish_list, list)
                assert len(fish_list) == 2
                assert all(isinstance(f, Fish) for f in fish_list)
                assert fish_list[0].id == 1
                assert fish_list[1].id == 2
                mock_get.assert_called_once_with("/api/v1/fish/")

    async def test_get_all_fish_empty(self):
        """Test getting all fish when none exist."""
        semaphore = asyncio.Semaphore(10)
        client = FishClient(
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
                fish_list = await client.get()
                assert fish_list is None

    async def test_post_fish(self):
        """Test creating a new fish."""
        semaphore = asyncio.Semaphore(10)
        client = FishClient(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        fish = Fish(id=None, species_id=100)
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = 1
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            async with client:
                fish_id = await client.post(fish)
                assert fish_id == 1
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                assert call_args[0][0] == "/api/v1/fish"

    async def test_post_measurement(self):
        """Test creating a new measurement for a fish."""
        semaphore = asyncio.Semaphore(10)
        client = FishClient(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        measurement = Measurement(id=None, fish_id=1, length_m=0.5, image_id=None)
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = 1
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            async with client:
                measurement_id = await client.post_measurement(1, measurement)
                assert measurement_id == 1
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                assert call_args[0][0] == "/api/v1/fish/1/measurements"

    async def test_get_species_by_scientific_name(self):
        """Test getting species by scientific name."""
        semaphore = asyncio.Semaphore(10)
        client = FishClient(
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
            "scientific_name": "Thunnus albacares",
            "common_name": "Yellowfin Tuna",
        }
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with client:
                species = await client.get_species_by_scientific_name(
                    "Thunnus albacares"
                )
                assert isinstance(species, Species)
                assert species.scientific_name == "Thunnus albacares"
                mock_get.assert_called_once_with(
                    "/api/v1/fish/species/Thunnus albacares"
                )

    async def test_get_species_by_scientific_name_not_found(self):
        """Test getting species by scientific name when not found."""
        semaphore = asyncio.Semaphore(10)
        client = FishClient(
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
                species = await client.get_species_by_scientific_name(
                    "Unknown species"
                )
                assert species is None

    async def test_get_species_by_scientific_name_returns_none_on_404(self):
        """A 404 from GET /fish/species/{name} returns None instead of raising."""
        semaphore = asyncio.Semaphore(10)
        client = FishClient(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

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
                species = await client.get_species_by_scientific_name("Unknown")
                assert species is None

    async def test_post_species(self):
        """Test creating a new species."""
        semaphore = asyncio.Semaphore(10)
        client = FishClient(
            base_url="http://test.com",
            username="testuser",
            password="testpass",
            timeout=10,
            semaphore=semaphore,
        )

        species = Species(
            id=None,
            scientific_name="Thunnus albacares",
            common_name="Yellowfin Tuna",
        )
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = 1
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            async with client:
                species_id = await client.post_species(species)
                assert species_id == 1
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                assert call_args[0][0] == "/api/v1/fish/species"
