"""Client for interacting with dive-related endpoints of the Fishsense API."""

from typing import List

import httpx
from retry import retry

from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.dive import Dive


class DiveClient(ClientBase):
    # pylint: disable=too-few-public-methods
    """Client for interacting with dive-related endpoints of the Fishsense API."""

    def __init__(self, base_url: str, timeout: int):
        super().__init__(base_url, timeout)

    @retry(exceptions=httpx.HTTPStatusError, tries=3, delay=2, backoff=2)
    async def get(self, dive_id: int | None = None) -> Dive | List[Dive] | None:
        """Get a dive.

        Returns:
            Dive | List[Dive]: The dive(s) retrieved from the API.
        """
        async with self._create_client() as client:
            if dive_id is not None:
                response = await client.get(f"/api/v1/dives/{dive_id}")
                response.raise_for_status()

                json = response.json()
                if json is None:
                    return None

                return Dive.model_validate(json)

            response = await client.get("/api/v1/dives/")
            response.raise_for_status()

            json = response.json()
            if json is None:
                return None

            return [Dive.model_validate(dive) for dive in json]
