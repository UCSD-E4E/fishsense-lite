"""Client for interacting with label-related endpoints of the Fishsense API."""

from typing import List

from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
from fishsense_api_sdk.models.headtail_label import HeadTailLabel
from fishsense_api_sdk.models.label_studio_sync_cursor import LabelStudioSyncCursor
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_sdk.models.species_label import SpeciesLabel


class LabelClient(ClientBase):
    """Client for interacting with label-related endpoints of the Fishsense API."""

    async def get_dive_slate_label(
        self,
        image_id: int | None = None,
        label_studio_id: int | None = None,
    ) -> DiveSlateLabel | None:
        """Get a DiveSlateLabel by image ID or Label Studio task ID.

        Args:
            image_id (int | None): The ID of the image to retrieve the slate label for.
            label_studio_id (int | None): The Label Studio task ID to retrieve the
                slate label for.

        Returns:
            DiveSlateLabel | None: The slate label, or None if not found.
        """
        if image_id is not None:
            response = await self._get(f"/api/v1/labels/dive-slate/{image_id}")
            if response.status_code == 404:
                self.logger.debug(
                    "No dive slate label found for image ID %s", image_id
                )
                return None
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug(
                    "No dive slate label found for image ID %s", image_id
                )
                return None

            return DiveSlateLabel.model_validate(json)

        if label_studio_id is not None:
            response = await self._get(
                f"/api/v1/labels/dive-slate/label-studio/{label_studio_id}"
            )
            if response.status_code == 404:
                self.logger.debug(
                    "No dive slate label found for label studio ID %s",
                    label_studio_id,
                )
                return None
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug(
                    "No dive slate label found for label studio ID %s",
                    label_studio_id,
                )
                return None

            return DiveSlateLabel.model_validate(json)

        raise NotImplementedError("Fetching without a parameter is not supported")

    async def get_dive_slate_labels(self, dive_id: int) -> List[DiveSlateLabel] | None:
        """Get dive slate labels for all images in a dive .

        Args:
            dive_id (int): The ID of the dive to retrieve dive slate labels for.

        Returns:
            List[DiveSlateLabel] | None: The list of dive slate labels for the specified dive.
        """
        response = await self._get(f"/api/v1/dives/{dive_id}/labels/dive-slate")
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug("No dive slate labels found for dive ID %s", dive_id)
            return None

        return [DiveSlateLabel.model_validate(label) for label in json]

    async def put_dive_slate_label(
        self, image_id: int, dive_slate_label: DiveSlateLabel
    ) -> int:
        """Put a dive slate labels to an image .

        Args:
            image_id (int): The ID of the image to put the dive slate labels to.
            dive_slate_labels (DiveSlateLabels): The dive slate labels to put.
        Returns:
            int: The ID of the created dive slate labels.
        """
        response = await self._put(
            f"/api/v1/labels/dive-slate/{image_id}",
            json=dive_slate_label.model_dump(exclude_unset=True, mode="json"),
        )
        response.raise_for_status()
        return response.json()

    async def get_headtail_label(
        self, image_id: int | None = None, label_studio_id: int | None = None
    ) -> HeadTailLabel | None:
        """Get a HeadTailLabel by its ID .

        Args:
            image_id (int): The ID of the image to retrieve the head-tail label for.

        Returns:
            HeadTailLabel | None: The head-tail label for the specified image.
        """
        if image_id is not None:
            response = await self._get(f"/api/v1/labels/headtail/{image_id}")
            if response.status_code == 404:
                self.logger.debug("No head-tail label found for image ID %s", image_id)
                return None
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug("No head-tail label found for image ID %s", image_id)
                return None

            return HeadTailLabel.model_validate(json)

        if label_studio_id is not None:
            response = await self._get(
                f"/api/v1/labels/headtail/label-studio/{label_studio_id}"
            )
            if response.status_code == 404:
                self.logger.debug(
                    "No head-tail label found for label studio ID %s", label_studio_id
                )
                return None
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug(
                    "No head-tail label found for label studio ID %s", label_studio_id
                )
                return None

            return HeadTailLabel.model_validate(json)

        raise NotImplementedError("Fetching without a parameter is not supported")

    async def get_headtail_labels(self, dive_id: int) -> List[HeadTailLabel] | None:
        """Get head-tail labels for all images in a dive .

        Args:
            dive_id (int): The ID of the dive to retrieve head-tail labels for.

        Returns:
            List[HeadTailLabel] | None: The list of head-tail labels for the specified dive.
        """
        response = await self._get(f"/api/v1/dives/{dive_id}/labels/headtail")
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug("No head-tail labels found for dive ID %s", dive_id)
            return None

        return [HeadTailLabel.model_validate(label) for label in json]

    async def get_headtail_label_studio_project_ids(
        self, *, incomplete: bool = False
    ) -> List[int]:
        """Get the distinct Label Studio project IDs that have head-tail labels.

        Single-query alternative to walking every canonical dive.

        Args:
            incomplete: when True, restrict to projects with at least one
                label whose `completed` is NULL or false.
        """
        suffix = "?incomplete=true" if incomplete else ""
        response = await self._get(
            f"/api/v1/labels/headtail/label-studio-project-ids{suffix}"
        )
        response.raise_for_status()
        return list(response.json() or [])

    async def get_species_label_studio_project_ids(
        self, *, incomplete: bool = False
    ) -> List[int]:
        """Get the distinct Label Studio project IDs that have species labels.

        Single-query alternative to walking every canonical dive.

        Args:
            incomplete: when True, restrict to projects with at least one
                label whose `completed` is NULL or false.
        """
        suffix = "?incomplete=true" if incomplete else ""
        response = await self._get(
            f"/api/v1/labels/species/label-studio-project-ids{suffix}"
        )
        response.raise_for_status()
        return list(response.json() or [])

    async def get_dive_slate_label_studio_project_ids(
        self, *, incomplete: bool = False
    ) -> List[int]:
        """Get the distinct Label Studio project IDs that have dive-slate labels.

        Single-query alternative to walking every canonical dive.

        Args:
            incomplete: when True, restrict to projects with at least one
                label whose `completed` is NULL or false.
        """
        suffix = "?incomplete=true" if incomplete else ""
        response = await self._get(
            f"/api/v1/labels/dive-slate/label-studio-project-ids{suffix}"
        )
        response.raise_for_status()
        return list(response.json() or [])

    async def put_headtail_label(
        self, image_id: int, headtail_label: HeadTailLabel
    ) -> int:
        """Put a head-tail label to an image .

        Args:
            image_id (int): The ID of the image to put the head-tail label to.
            headtail_label (HeadTailLabel): The head-tail label to put.
        Returns:
            int: The ID of the created head-tail label.
        """
        response = await self._put(
            f"/api/v1/labels/headtail/{image_id}",
            json=headtail_label.model_dump(exclude_unset=True, mode="json"),
        )
        response.raise_for_status()
        return response.json()

    async def get_laser_label(
        self, image_id: int | None = None, label_studio_id: int | None = None
    ) -> LaserLabel | None:
        """Get a LaserLabel by its ID .

        Args:
            image_id (int | None): The ID of the image to retrieve the laser label for.
            label_studio_id (int | None): ID of the label studio entry to find the laser label for.

        Returns:
            LaserLabel | None: The laser label for the specified image.
        """
        if image_id is not None:
            response = await self._get(f"/api/v1/labels/laser/{image_id}")
            if response.status_code == 404:
                self.logger.debug("No laser label found for image ID %s", image_id)
                return None
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug("No laser label found for image ID %s", image_id)
                return None

            return LaserLabel.model_validate(json)

        if label_studio_id is not None:
            response = await self._get(
                f"/api/v1/labels/laser/label-studio/{label_studio_id}"
            )
            if response.status_code == 404:
                self.logger.debug(
                    "No laser label found for label studio ID %s", label_studio_id
                )
                return None
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug(
                    "No laser label found for label studio ID %s", label_studio_id
                )
                return None

            return LaserLabel.model_validate(json)

        raise NotImplementedError("Fetching without a parameter is not supported")

    async def get_laser_labels(self, dive_id: int) -> List[LaserLabel] | None:
        """Get laser labels for all images in a dive .

        Args:
            dive_id (int): The ID of the dive to retrieve laser labels for.

        Returns:
            List[LaserLabel] | None: The list of laser labels for the specified dive.
        """
        response = await self._get(f"/api/v1/dives/{dive_id}/labels/laser")
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug("No laser labels found for dive ID %s", dive_id)
            return None

        return [LaserLabel.model_validate(label) for label in json]

    async def get_laser_label_studio_project_ids(
        self, *, incomplete: bool = False
    ) -> List[int]:
        """Get the distinct Label Studio project IDs that have laser labels.

        Single-query alternative to walking every canonical dive.

        Args:
            incomplete: when True, restrict to projects with at least one
                label whose `completed` is NULL or false.
        """
        suffix = "?incomplete=true" if incomplete else ""
        response = await self._get(
            f"/api/v1/labels/laser/label-studio-project-ids{suffix}"
        )
        response.raise_for_status()
        return list(response.json() or [])

    async def get_dives_with_complete_laser_labeling(self) -> List[int]:
        """Get dive IDs whose laser labeling is fully complete.

        A dive qualifies iff every non-superseded `LaserLabel` on its
        images has `completed=True` and at least one such label exists.
        """
        response = await self._get(
            "/api/v1/labels/laser/dives-with-complete-labeling"
        )
        response.raise_for_status()
        return list(response.json() or [])

    async def put_laser_label(self, image_id: int, laser_label: LaserLabel) -> int:
        """Put a laser label to an image .

        Args:
            image_id (int): The ID of the image to put the laser label to.
            laser_label (LaserLabel): The laser label to put.
        Returns:
            int: The ID of the created laser label.
        """
        response = await self._put(
            f"/api/v1/labels/laser/{image_id}",
            json=laser_label.model_dump(exclude_unset=True, mode="json"),
        )
        response.raise_for_status()
        return response.json()

    async def get_species_label(
        self,
        image_id: int | None = None,
        label_studio_id: int | None = None,
    ) -> SpeciesLabel | None:
        """Get a SpeciesLabel by image ID or Label Studio task ID.

        Args:
            image_id (int | None): The ID of the image to retrieve the species
                label for.
            label_studio_id (int | None): The Label Studio task ID to retrieve
                the species label for.

        Returns:
            SpeciesLabel | None: The species label, or None if not found.
        """
        if image_id is not None:
            response = await self._get(f"/api/v1/labels/species/{image_id}")
            if response.status_code == 404:
                self.logger.debug("No species label found for image ID %s", image_id)
                return None
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug("No species label found for image ID %s", image_id)
                return None

            return SpeciesLabel.model_validate(json)

        if label_studio_id is not None:
            response = await self._get(
                f"/api/v1/labels/species/label-studio/{label_studio_id}"
            )
            if response.status_code == 404:
                self.logger.debug(
                    "No species label found for label studio ID %s",
                    label_studio_id,
                )
                return None
            response.raise_for_status()

            json = response.json()
            if json is None:
                self.logger.debug(
                    "No species label found for label studio ID %s",
                    label_studio_id,
                )
                return None

            return SpeciesLabel.model_validate(json)

        raise NotImplementedError("Fetching without a parameter is not supported")

    async def get_species_labels(self, dive_id: int) -> List[SpeciesLabel] | None:
        """Get species labels for all images in a dive .

        Args:
            dive_id (int): The ID of the dive to retrieve species labels for.

        Returns:
            List[SpeciesLabel] | None: The list of species labels for the specified dive.
        """
        response = await self._get(f"/api/v1/dives/{dive_id}/labels/species")
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug("No species labels found for dive ID %s", dive_id)
            return None

        return [SpeciesLabel.model_validate(label) for label in json]

    async def put_species_label(
        self, image_id: int, species_label: SpeciesLabel
    ) -> int:
        """Put a species label to an image .

        Args:
            image_id (int): The ID of the image to put the species label to.
            species_label (SpeciesLabel): The species label to put.
        Returns:
            int: The ID of the created species label.
        """
        response = await self._put(
            f"/api/v1/labels/species/{image_id}",
            json=species_label.model_dump(exclude_unset=True, mode="json"),
        )
        response.raise_for_status()
        return response.json()

    async def get_sync_cursor(
        self, kind: str, label_studio_project_id: int
    ) -> LabelStudioSyncCursor | None:
        """Get the incremental-sync cursor for a (kind, project) pair.

        Returns None when the API has no cursor recorded yet — the
        api-workflow-worker treats that as "first run, sync everything."
        """
        response = await self._get(
            f"/api/v1/labels/sync-cursor/{kind}/{label_studio_project_id}"
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()

        json = response.json()
        if json is None:
            return None

        return LabelStudioSyncCursor.model_validate(json)

    async def put_sync_cursor(
        self,
        kind: str,
        label_studio_project_id: int,
        cursor: LabelStudioSyncCursor,
    ) -> int:
        """Upsert the incremental-sync cursor for a (kind, project) pair."""
        response = await self._put(
            f"/api/v1/labels/sync-cursor/{kind}/{label_studio_project_id}",
            json=cursor.model_dump(exclude_unset=True, mode="json"),
        )
        response.raise_for_status()
        return response.json()
