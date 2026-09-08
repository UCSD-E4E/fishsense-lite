"""Client for the calibration-target endpoints of the Fishsense API."""

from typing import List

from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.calibration_target import CalibrationTarget


class CalibrationTargetClient(ClientBase):
    # pylint: disable=too-few-public-methods
    """Client for the calibration-target endpoints of the Fishsense API."""

    async def get(self) -> List[CalibrationTarget] | None:
        """Get all calibration targets.

        Returns:
            List[CalibrationTarget] | None: The calibration targets.
        """
        response = await self._get("/api/v1/calibration-targets/")
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug("No calibration targets found.")
            return None

        return [CalibrationTarget.model_validate(target) for target in json]

    # @app.put("/api/v1/calibration-targets/{calibration_target_id}", status_code=201)
    async def put(self, calibration_target: CalibrationTarget) -> int:
        """Create or update a calibration target.

        Args:
            calibration_target (CalibrationTarget): The target to put.

        Returns:
            int: The ID of the calibration target.
        """
        response = await self._put(
            f"/api/v1/calibration-targets/{calibration_target.id}",
            json=calibration_target.model_dump(exclude_unset=True, mode="json"),
        )
        response.raise_for_status()

        return response.json()
