""" "Client for fish-related endpoints in the FishSense API SDK."""

from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.fish import Fish
from fishsense_api_sdk.models.measurement import Measurement
from fishsense_api_sdk.models.species import Species


class FishClient(ClientBase):
    """Client for interacting with fish-related endpoints in the FishSense API."""

    async def get(self, fish_id: int | None = None) -> list[Fish] | Fish | None:
        """Get a list of fish objects .

        Returns:
            List[Fish] | Fish | None: The fish object(s) retrieved from the API.
        """
        if fish_id is not None:
            response = await self._get(f"/api/v1/fish/{fish_id}")
            if response.status_code == 404:
                self.logger.debug("No fish found with ID %s", fish_id)
                return None
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug("No fish found with ID %s", fish_id)
                return None

            return Fish.model_validate(json)

        response = await self._get("/api/v1/fish/")
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug("No fish found.")
            return None

        return [Fish.model_validate(fish) for fish in json]

    async def post(self, fish: Fish) -> int:
        """Create a new fish entry in the Fishsense API.

        Args:
            fish (Fish): The fish data to create.

        Returns:
            int: The ID of the created fish.
        """
        response = await self._post(
            "/api/v1/fish",
            json=fish.model_dump(),
        )
        response.raise_for_status()
        return response.json()

    async def get_measurements(self, dive_id: int) -> list[Measurement] | None:
        """Get every measurement recorded for the images in a dive.

        Stage 14 reads this once per dive so it can skip images it has
        already measured — `post_measurement` is a create, so without
        this filter a re-run on a partially-measured dive would record
        the same fish twice.

        Args:
            dive_id (int): The ID of the dive to retrieve measurements for.

        Returns:
            list[Measurement] | None: The measurements for the dive, or
            None when the dive has none yet.
        """
        response = await self._get(f"/api/v1/dives/{dive_id}/measurements")
        if response.status_code == 404:
            self.logger.debug("No measurements found for dive ID %s", dive_id)
            return None
        response.raise_for_status()
        json = response.json()
        if json is None:
            return None
        return [Measurement.model_validate(m) for m in json]

    async def post_measurement(self, fish_id: int, measurement: Measurement) -> int:
        """Create a new measurement entry for a fish in the Fishsense API.

        Args:
            fish_id (int): The ID of the fish to associate the measurement with.
            length_m (float): The length measurement in meters.

        Returns:
            int: The ID of the created measurement.
        """
        response = await self._post(
            f"/api/v1/fish/{fish_id}/measurements",
            json=measurement.model_dump(),
        )
        response.raise_for_status()
        return response.json()

    async def get_species_by_scientific_name(
        self, scientific_name: str
    ) -> Species | None:
        """Retrieve species information by scientific name.

        Args:get_species_by_scientific_name
            scientific_name (str): The scientific name of the species to retrieve.

        Returns:
            Species | None: The species object if found, otherwise None.
        """
        response = await self._get(f"/api/v1/fish/species/{scientific_name}")
        if response.status_code == 404:
            self.logger.debug(
                "No species found with scientific name %s", scientific_name
            )
            return None
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug(
                "No species found with scientific name %s", scientific_name
            )
            return None

        return Species.model_validate(json)

    async def post_species(self, species: Species) -> int:
        """Create a new species entry in the Fishsense API.

        Args:
            species (Species): The species data to create.

        Returns:
            int: The ID of the created species.
        """
        response = await self._post(
            "/api/v1/fish/species",
            json=species.model_dump(),
        )
        response.raise_for_status()
        return response.json()
