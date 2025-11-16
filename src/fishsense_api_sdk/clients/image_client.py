"""Client for interacting with image-related endpoints of the Fishsense API."""

from typing import List

from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.dive_frame_cluster import DiveFrameCluster
from fishsense_api_sdk.models.image import Image


class ImageClient(ClientBase):
    # pylint: disable=too-few-public-methods
    """Client for interacting with image-related endpoints of the Fishsense API."""

    def __init__(self, base_url: str, timeout: int):
        super().__init__(base_url, timeout)

    async def get(
        self, dive_id: int | None = None, image_id: int | None = None
    ) -> Image | List[Image] | None:
        """Get images from dive .

        Raises:
            NotImplementedError: Getting all images is not supported currently.

        Returns:
            Image | List[Image]: The image(s) retrieved from the API.
        """
        async with self._create_client() as client:
            if dive_id is not None:
                response = await client.get(f"/api/v1/dives/{dive_id}/images/")
                response.raise_for_status()

                json = response.json()
                if json is None:
                    return None

                return [Image.model_validate(image) for image in json]

            if image_id is not None:
                response = await client.get(f"/api/v1/images/{image_id}")
                response.raise_for_status()

                json = response.json()
                if json is None:
                    return None

                return Image.model_validate(json)

            raise NotImplementedError("Getting all images is not supported.")

    async def get_clusters(self, dive_id: int) -> List[DiveFrameCluster] | None:
        """Get clusters of images in the dive_id.

        Args:
            dive_id (int): The ID of the dive to retrieve clusters for.

        Returns:
            List[DiveFrameCluster]: The list of image clusters for the specified dive.
        """
        async with self._create_client() as client:
            response = await client.get(f"/api/v1/dives/{dive_id}/images/clusters/")
            response.raise_for_status()

            json = response.json()
            if json is None:
                return None

            return [DiveFrameCluster.model_validate(cluster) for cluster in json]

    async def post_cluster(self, dive_id: int, image_ids: List[int]) -> int:
        """Insert images in the dive cluster .

        Args:
            dive_id (int): The ID of the dive to insert images into.
            image_ids (List[int]): The IDs of the images to insert.

        Returns:
            int: The ID of the created dive frame cluster.
        """
        async with self._create_client() as client:
            response = await client.post(
                f"/api/v1/dives/{dive_id}/images/clusters/",
                json=image_ids,
            )
            response.raise_for_status()
            return response.json()
