"""Client for interacting with image-related endpoints of the Fishsense API."""

from typing import Any, Dict, List

from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.dive_frame_cluster import DiveFrameCluster
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_depth import LaserDepth


class ImageClient(ClientBase):
    """Client for interacting with image-related endpoints of the Fishsense API."""

    async def get(
        self,
        dive_id: int | None = None,
        image_id: int | None = None,
        checksum: str | None = None,
    ) -> Image | List[Image] | None:
        # pylint: disable=too-many-return-statements
        # Three lookup modes x (404 -> None, null body -> None, hit) is
        # inherently branchy; splitting it would change the public surface.
        """Get images from dive .

        Raises:
            NotImplementedError: Getting all images is not supported currently.

        Returns:
            Image | List[Image]: The image(s) retrieved from the API.
        """
        if dive_id is not None:
            response = await self._get(f"/api/v1/dives/{dive_id}/images/")
            if response.status_code == 404:
                # The endpoint 404s an *empty* dive rather than returning [],
                # and that is the normal state of a dive ingest has just
                # created. Raising here killed the first scan of every new
                # dive before it wrote a single row.
                self.logger.debug("No images found for dive ID %s", dive_id)
                return None
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug("No images found for dive ID %s", dive_id)
                return None

            return [Image.model_validate(image) for image in json]

        if image_id is not None:
            response = await self._get(f"/api/v1/images/{image_id}")
            if response.status_code == 404:
                self.logger.debug("No image found with ID %s", image_id)
                return None
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug("No image found with ID %s", image_id)
                return None

            return Image.model_validate(json)

        if checksum is not None:
            response = await self._get(f"/api/v1/images/checksum/{checksum}")
            if response.status_code == 404:
                self.logger.debug("No image found with checksum %s", checksum)
                return None
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug("No image found with checksum %s", checksum)
                return None

            return Image.model_validate(json)

        raise NotImplementedError("Getting all images is not supported.")

    async def post(
        self, dive_id: int, image: Image, *, set_canonical: bool = False
    ) -> int:
        """Create or update an image within a dive, keyed on its `path`.

        **`is_canonical` is stripped from the payload by default.** The server
        computes it -- the first row for a given checksum is canonical, later
        duplicates are not -- and an explicit value in the body overrides that.
        Since `Image.is_canonical` is a required field on this model, it always
        carries *some* value, so `exclude_unset` alone cannot keep it off the
        wire; posting it unconditionally would mark every duplicate frame
        canonical and silently destroy the distinction that lets the same
        physical frames live under two dive rows (prod dives 64 and 66).

        Pass `set_canonical=True` only to deliberately override the server --
        e.g. promoting a re-ingested copy after the original dive's files were
        lost.

        Args:
            dive_id (int): The dive the image belongs to.
            image (Image): The image to create or update.
            set_canonical (bool): Send `is_canonical` and override the
                server-side computation. Defaults to False.

        Returns:
            int: The ID of the created or updated image.
        """
        payload = image.model_dump(exclude_unset=True, mode="json")
        if not set_canonical:
            payload.pop("is_canonical", None)

        response = await self._post(
            f"/api/v1/dives/{dive_id}/images/",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def lookup_checksums(
        self, checksums: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Look up which dives already hold each of `checksums`.

        Returns `{checksum: [{image_id, dive_id, is_canonical}, ...]}`, with an
        empty list for checksums that aren't known. Used at ingest time to
        report content overlap with existing dives -- e.g. "48 of 55 frames
        already exist in dive 64, so those will land non-canonical" -- before
        the dive is committed.

        Args:
            checksums (List[str]): MD5 hexdigests to look up.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Per-checksum hits.
        """
        response = await self._post(
            "/api/v1/images/checksums/lookup",
            json=list(checksums),
        )
        response.raise_for_status()
        return response.json()

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

    async def get_laser_depth(self, image_id: int) -> LaserDepth | None:
        """Get the distance to an image's laser dot.

        Args:
            image_id (int): The image to retrieve the depth for.

        Returns:
            LaserDepth | None: The depth, or None when none has been computed
                for this image yet.
        """
        response = await self._get(f"/api/v1/images/{image_id}/laser-depth/")
        if response.status_code == 404:
            self.logger.debug("No laser depth found for image ID %s", image_id)
            return None
        response.raise_for_status()
        return LaserDepth.model_validate(response.json())

    async def put_laser_depth(self, image_id: int, depth: LaserDepth) -> int:
        """Upsert the distance to an image's laser dot.

        Args:
            image_id (int): The image the depth belongs to.
            depth (LaserDepth): The computed depth, carrying the laser label
                and calibration it was derived from.

        Returns:
            int: The id of the upserted depth row.
        """
        response = await self._put(
            f"/api/v1/images/{image_id}/laser-depth/",
            json=depth.model_dump(exclude_unset=True, mode="json"),
        )
        response.raise_for_status()
        return response.json()

    async def get_laser_depths(self, dive_id: int) -> List[LaserDepth]:
        """Get every laser depth recorded for a dive's images.

        Empty list when the dive has none — the compute stage reads this once
        per dive to see which images it has already done, and "none yet" is
        the normal first-run state.

        Args:
            dive_id (int): The dive to retrieve depths for.

        Returns:
            List[LaserDepth]: The depths for the dive's images.
        """
        response = await self._get(f"/api/v1/dives/{dive_id}/laser-depths/")
        response.raise_for_status()
        return [LaserDepth.model_validate(row) for row in (response.json() or [])]
