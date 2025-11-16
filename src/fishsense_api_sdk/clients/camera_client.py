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

    def __init__(self, base_url: str, timeout: int):
        super().__init__(base_url, timeout)

    async def get(self, camera_id: int | None = None) -> List[Camera] | Camera | None:
        """Get a list of camera objects .

        Returns:
            List[Camera] | Camera | None: The camera object(s) retrieved from the API.
        """
        async with self._create_client() as client:
            if camera_id is not None:
                response = await client.get(f"/api/v1/cameras/{camera_id}")
                response.raise_for_status()

                json = response.json()
                if json is None:
                    return None

                return Camera.model_validate(json)

            response = await client.get("/api/v1/cameras/")
            response.raise_for_status()

            json = response.json()
            if json is None:
                return None

            return [Camera.model_validate(camera) for camera in json]

    async def get_intrinsics(self, camera_id: int) -> CameraIntrinsics | None:
        """Returns the intrinsic intrinsics for a camera .

        Args:
            camera_id (int): The ID of the camera to retrieve intrinsics for.

        Returns:
            CameraIntrinsics: The intrinsic parameters of the specified camera.
        """
        async with self._create_client() as client:
            response = await client.get(f"/api/v1/cameras/{camera_id}/intrinsics/")
            response.raise_for_status()

            json = response.json()
            if json is None:
                return None

            return CameraIntrinsics._from_internal(  # pylint: disable=protected-access
                _CameraIntrinsics.model_validate(json)
            )

    async def post_intrinsics(
        self, camera_id: int, camera_intrinsics: CameraIntrinsics
    ) -> int:
        """Post the intrinsic intrinsics to a camera .

        Args:
            camera_id (int): The ID of the camera to set intrinsics for.
            camera_intrinsics (CameraIntrinsics): The intrinsic parameters to set for the camera.

        Returns:
            int: The ID of the camera with updated intrinsics.
        """
        async with self._create_client() as client:
            response = await client.post(
                f"/api/v1/cameras/{camera_id}/intrinsics/",
                json=camera_intrinsics._to_internal().model_dump(),  # pylint: disable=protected-access
            )
            response.raise_for_status()
            return response.json()
