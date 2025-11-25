"""Client for interacting with dive-related endpoints of the Fishsense API."""

from typing import List

from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.dive import Dive


class DiveClient(ClientBase):
    # pylint: disable=too-few-public-methods
    """Client for interacting with dive-related endpoints of the Fishsense API."""

    async def get(self, dive_id: int | None = None) -> Dive | List[Dive] | None:
        """Get a dive.

        Returns:
            Dive | List[Dive]: The dive(s) retrieved from the API.
        """
        if dive_id is not None:
            response = await self._get(f"/api/v1/dives/{dive_id}")
            response.raise_for_status()

            json = response.json()
            if json is None:
                return None

            return Dive.model_validate(json)

        response = await self._get("/api/v1/dives/")
        response.raise_for_status()

        json = response.json()
        if json is None:
            return None

        return [Dive.model_validate(dive) for dive in json]

    async def get_canonical(self) -> List[Dive] | None:
        """Get canonical dives.

        Returns:
            List[Dive] | None: The canonical dives retrieved from the API.
        """
        response = await self._get("/api/v1/canonical/dives/")
        response.raise_for_status()

        json = response.json()
        if json is None:
            return None

        return [Dive.model_validate(dive) for dive in json]
