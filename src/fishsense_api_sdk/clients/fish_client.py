from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.species import Species


class FishClient(ClientBase):
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
        response.raise_for_status()

        json = response.json()
        if json is None:
            return None

        return Species.model_validate(json)

    async def post(self, species: Species) -> int:
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
