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
            if response.status_code == 404:
                self.logger.debug("No dive found with ID %s", dive_id)
                return None
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug("No dive found with ID %s", dive_id)
                return None

            return Dive.model_validate(json)

        response = await self._get("/api/v1/dives/")
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug("No dives found.")
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
            self.logger.debug("No canonical dives found.")
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
        if response.status_code == 404:
            self.logger.debug("No laser extrinsics found for dive ID %s", dive_id)
            return None
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug("No laser extrinsics found for dive ID %s", dive_id)
            return None

        return LaserExtrinsics._from_internal(  # pylint: disable=protected-access
            _LaserExtrinsics.model_validate(json)
        )

    async def select_next_for_laser_preprocessing(self) -> int | None:
        """Stage 0.1 cohort selector: returns the next HIGH-priority
        dive needing laser preprocessing, or None when the cohort is
        empty. Server-side single-query equivalent of the api-worker's
        `select_next_high_priority_dive_for_laser_preprocessing_activity`.
        """
        return await self._select_next("laser-preprocessing")

    async def select_next_for_dive_image_preprocessing(self) -> int | None:
        """Stage 2 cohort selector. See `select_next_for_laser_preprocessing`."""
        return await self._select_next("dive-image-preprocessing")

    async def select_next_for_headtail_preprocessing(self) -> int | None:
        """Stage 5.1 cohort selector. See `select_next_for_laser_preprocessing`."""
        return await self._select_next("headtail-preprocessing")

    async def select_next_for_slate_preprocessing(self) -> int | None:
        """Stage 9 cohort selector. See `select_next_for_laser_preprocessing`."""
        return await self._select_next("slate-preprocessing")

    async def select_next_for_laser_calibration(self) -> int | None:
        """Stage 13 cohort selector. See `select_next_for_laser_preprocessing`."""
        return await self._select_next("laser-calibration")

    async def select_next_for_measure_fish(self) -> int | None:
        """Stage 14 cohort selector. See `select_next_for_laser_preprocessing`."""
        return await self._select_next("measure-fish")

    async def _select_next(self, cohort: str) -> int | None:
        response = await self._get(f"/api/v1/dives/select-next/{cohort}/")
        response.raise_for_status()
        return response.json()

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

        return response.json()
