"""Client for interacting with label-related endpoints of the Fishsense API."""

from typing import List

import httpx
from retry import retry

from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_sdk.models.species_label import SpeciesLabel


class LabelClient(ClientBase):
    """Client for interacting with label-related endpoints of the Fishsense API."""

    def __init__(self, base_url: str, timeout: int):
        super().__init__(base_url, timeout)

    async def get_laser_label(self, image_id: int) -> LaserLabel | None:
        """Get a LaserLabel by its ID .

        Args:
            image_id (int): The ID of the image to retrieve the laser label for.

        Returns:
            LaserLabel | None: The laser label for the specified image.
        """
        async with self._create_client() as client:
            response = await client.get(f"/api/v1/labels/laser/{image_id}")
            response.raise_for_status()

            json = response.json()
            if json is None:
                return None

            return LaserLabel.model_validate(json)

    @retry(exceptions=httpx.HTTPStatusError, tries=3, delay=2, backoff=2)
    @retry(exceptions=httpx.ConnectError, tries=3, delay=2, backoff=2)
    async def get_laser_labels(self, dive_id: int) -> List[LaserLabel] | None:
        """Get laser labels for all images in a dive .

        Args:
            dive_id (int): The ID of the dive to retrieve laser labels for.

        Returns:
            List[LaserLabel] | None: The list of laser labels for the specified dive.
        """
        async with self._create_client() as client:
            response = await client.get(f"/api/v1/dives/{dive_id}/labels/laser")
            response.raise_for_status()

            json = response.json()
            if json is None:
                return None

            return [LaserLabel.model_validate(label) for label in json]

    async def get_species_label(self, image_id: int) -> SpeciesLabel | None:
        """Get a species label .

        Args:
            image_id (int): The ID of the image to retrieve the species label for.

        Returns:
            SpeciesLabel | None: The species label for the specified image.
        """
        async with self._create_client() as client:
            response = await client.get(f"/api/v1/labels/species/{image_id}")
            response.raise_for_status()

            json = response.json()
            if json is None:
                return None

            return SpeciesLabel.model_validate(json)

    async def get_species_labels(self, dive_id: int) -> List[SpeciesLabel] | None:
        """Get species labels for all images in a dive .

        Args:
            dive_id (int): The ID of the dive to retrieve species labels for.

        Returns:
            List[SpeciesLabel] | None: The list of species labels for the specified dive.
        """
        async with self._create_client() as client:
            response = await client.get(f"/api/v1/dives/{dive_id}/labels/species")
            response.raise_for_status()

            json = response.json()
            if json is None:
                return None

            return [SpeciesLabel.model_validate(label) for label in json]

    async def put_species_label(
        self, image_id: int, species_label: SpeciesLabel
    ) -> int:
        """Put a species label to an image .

        Args:
            image_id (int): The ID of the image to put the species label to.
            species_label (SpeciesLabel): The species label to put.
        Returns:
            int: The ID of the created species label.
        """
        async with self._create_client() as client:
            response = await client.put(
                f"/api/v1/labels/species/{image_id}",
                json=species_label.model_dump(exclude_unset=True, mode="json"),
            )
            response.raise_for_status()
            return response.json()
