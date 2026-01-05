"""Client for interacting with camera-related endpoints of the Fishsense API."""

from typing import List

from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.camera import Camera
from fishsense_api_sdk.models.camera_intrinsics import (
    CameraIntrinsics,
    _CameraIntrinsics,
)


class CameraClient(ClientBase):
    """Client for interacting with camera-related endpoints of the Fishsense API."""

    async def get(self, camera_id: int | None = None) -> List[Camera] | Camera | None:
        """Get a list of camera objects .

        Returns:
            List[Camera] | Camera | None: The camera object(s) retrieved from the API.
        """
        if camera_id is not None:
            response = await self._get(f"/api/v1/cameras/{camera_id}")
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug("No camera found with ID %s", camera_id)
                return None

            return Camera.model_validate(json)

        response = await self._get("/api/v1/cameras/")
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug("No cameras found.")
            return None

        return [Camera.model_validate(camera) for camera in json]

    async def get_intrinsics(self, camera_id: int) -> CameraIntrinsics | None:
        """Returns the intrinsic intrinsics for a camera .

        Args:
            camera_id (int): The ID of the camera to retrieve intrinsics for.

        Returns:
            CameraIntrinsics: The intrinsic parameters of the specified camera.
        """
        response = await self._get(f"/api/v1/cameras/{camera_id}/intrinsics/")
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug("No intrinsics found for camera ID %s", camera_id)
            return None

        return CameraIntrinsics._from_internal(  # pylint: disable=protected-access
            _CameraIntrinsics.model_validate(json)
        )

    async def put_intrinsics(
        self, camera_id: int, camera_intrinsics: CameraIntrinsics
    ) -> int:
        """Post the intrinsic intrinsics to a camera .

        Args:
            camera_id (int): The ID of the camera to set intrinsics for.
            camera_intrinsics (CameraIntrinsics): The intrinsic parameters to set for the camera.

        Returns:
            int: The ID of the camera with updated intrinsics.
        """
        response = await self._put(
            f"/api/v1/cameras/{camera_id}/intrinsics/",
            json=camera_intrinsics._to_internal().model_dump(),  # pylint: disable=protected-access
        )
        response.raise_for_status()
        return response.json()
