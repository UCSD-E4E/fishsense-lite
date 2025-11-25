"""Client for interacting with dive-related endpoints of the Fishsense API."""

from typing import List

from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.laser_extrinsics import LaserExtrinsics, _LaserExtrinsics


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

    async def get_laser_extrinsics(self, dive_id: int) -> LaserExtrinsics | None:
        """Get laser extrinsics for a dive.

        Args:
            dive_id (int): The ID of the dive to retrieve laser extrinsics for.

        Returns:
            LaserExtrinsics | None: The laser extrinsics of the specified dive.
        """
        response = await self._get(f"/api/v1/dives/{dive_id}/laser-extrinsics/")
        response.raise_for_status()

        json = response.json()
        if json is None:
            return None

        return LaserExtrinsics._from_internal(  # pylint: disable=protected-access
            _LaserExtrinsics.model_validate(json)
        )

    async def put_laser_extrinsics(
        self, dive_id: int, laser_extrinsics: LaserExtrinsics
    ) -> int:
        """Put laser extrinsics for a dive.

        Args:
            dive_id (int): The ID of the dive to set laser extrinsics for.
            laser_extrinsics (LaserExtrinsics): The laser extrinsics to set for the dive.

        Returns:
            int: The ID of the dive with updated laser extrinsics.
        """
        response = await self._put(
            f"/api/v1/dives/{dive_id}/laser-extrinsics/",
            json=laser_extrinsics._to_internal().model_dump(  # pylint: disable=protected-access
                exclude_unset=True, mode="json"
            ),
        )
        response.raise_for_status()

        response.raise_for_status()
        return response.json()
