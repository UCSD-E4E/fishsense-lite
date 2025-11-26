from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.species import Species


class FishClient(ClientBase):
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
