"""Client for interacting with image-related endpoints of the Fishsense API."""

from typing import List

from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.dive_frame_cluster import DiveFrameCluster
from fishsense_api_sdk.models.image import Image


class ImageClient(ClientBase):
    # pylint: disable=too-few-public-methods
    """Client for interacting with image-related endpoints of the Fishsense API."""

    async def get(
        self,
        dive_id: int | None = None,
        image_id: int | None = None,
        checksum: str | None = None,
    ) -> Image | List[Image] | None:
        """Get images from dive .

        Raises:
            NotImplementedError: Getting all images is not supported currently.

        Returns:
            Image | List[Image]: The image(s) retrieved from the API.
        """
        if dive_id is not None:
            response = await self._get(f"/api/v1/dives/{dive_id}/images/")
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug("No images found for dive ID %s", dive_id)
                return None

            return [Image.model_validate(image) for image in json]

        if image_id is not None:
            response = await self._get(f"/api/v1/images/{image_id}")
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug("No image found with ID %s", image_id)
                return None

            return Image.model_validate(json)

        if checksum is not None:
            response = await self._get(f"/api/v1/images/checksum/{checksum}")
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug("No image found with checksum %s", checksum)
                return None

            return Image.model_validate(json)

        raise NotImplementedError("Getting all images is not supported.")

    async def get_clusters(
        self, dive_id: int, data_source: str
    ) -> List[DiveFrameCluster] | None:
        """Get clusters of images in the dive_id.

        Args:
            dive_id (int): The ID of the dive to retrieve clusters for.

        Returns:
            List[DiveFrameCluster]: The list of image clusters for the specified dive.
        """
        response = await self._get(
            f"/api/v1/dives/{dive_id}/images/clusters/{data_source}"
        )
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug(
                "No image clusters found for dive ID %s and data source %s",
                dive_id,
                data_source,
            )
            return None

        return [DiveFrameCluster.model_validate(cluster) for cluster in json]

    async def post_cluster(
        self, dive_id: int, dive_frame_cluster: DiveFrameCluster
    ) -> int:
        """Insert images in the dive cluster .

        Args:
            dive_id (int): The ID of the dive to insert images into.
            image_ids (List[int]): The IDs of the images to insert.

        Returns:
            int: The ID of the created dive frame cluster.
        """
        dive_frame_cluster.dive_id = dive_id

        response = await self._post(
            f"/api/v1/dives/{dive_id}/images/clusters/",
            json=dive_frame_cluster.model_dump(exclude_unset=True, mode="json"),
        )
        response.raise_for_status()
        return response.json()

    async def put_cluster(
        self,
        dive_id: int,
        dive_frame_cluster_id: int,
        dive_frame_cluster: DiveFrameCluster,
    ) -> int:
        """Update images in the dive cluster .

        Args:
            dive_id (int): The ID of the dive to update images in.
            dive_frame_cluster_id (int): The ID of the dive frame cluster to update.
            image_ids (List[int]): The IDs of the images to update.

        Returns:
            int: The ID of the updated dive frame cluster.
        """
        dive_frame_cluster.dive_id = dive_id

        response = await self._put(
            f"/api/v1/dives/{dive_id}/images/clusters/{dive_frame_cluster_id}",
            json=dive_frame_cluster.model_dump(exclude_unset=True, mode="json"),
        )
        response.raise_for_status()
        return response.json()
