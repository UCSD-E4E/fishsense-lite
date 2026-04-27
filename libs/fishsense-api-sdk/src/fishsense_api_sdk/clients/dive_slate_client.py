"""Client for interacting with dive slate-related endpoints of the Fishsense API."""

from typing import List

from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.dive_slate import DiveSlate


class DiveSlateClient(ClientBase):
    # pylint: disable=too-few-public-methods
    """Client for interacting with dive slate-related endpoints of the Fishsense API."""

    async def get(self) -> List[DiveSlate] | None:
        """Get all dive slates.

        Returns:
            List[DiveSlate] | None: The list of dive slates.
        """
        response = await self._get("/api/v1/dive-slates/")
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug("No dive slates found.")
            return None

        return [DiveSlate.model_validate(dive_slate) for dive_slate in json]

    # @app.put("/api/v1/dive-slates/{dive_slate_id}", status_code=201)
    async def put(self, dive_slate: DiveSlate) -> int:
        """Put a dive slate.

        Args:
            dive_slate (DiveSlate): The dive slate to put.

        Returns:
            int: The ID of the created dive slate.
        """
        response = await self._put(
            f"/api/v1/dive-slates/{dive_slate.id}",
            json=dive_slate.model_dump(exclude_unset=True, mode="json"),
        )
        response.raise_for_status()

        return response.json()
