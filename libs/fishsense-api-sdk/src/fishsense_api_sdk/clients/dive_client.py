"""Client for interacting with dive-related endpoints of the Fishsense API."""

from typing import List
from urllib.parse import urlencode

from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.calibration_candidate import CalibrationCandidate
from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.dive_laser_line import DiveLaserLine
from fishsense_api_sdk.models.laser_extrinsics import LaserExtrinsics, _LaserExtrinsics


class DiveClient(ClientBase):
    # pylint: disable=too-few-public-methods,too-many-public-methods
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

    async def post(self, dive: Dive) -> int:
        """Create or update a dive, keyed on its NAS-relative `path`.

        The endpoint upserts on `path`, so there is no create-vs-update
        decision here: ingest posts a dive at `priority=LOW` when it first sees
        the folder, then posts the same path again once every image has landed
        to flip it to HIGH. Both calls come through this method.

        `mode="json"` is required -- `dive_datetime` is a `datetime`, which
        httpx cannot encode.

        Args:
            dive (Dive): The dive to create or update.

        Returns:
            int: The ID of the created or updated dive.
        """
        response = await self._post(
            "/api/v1/dives/",
            json=dive.model_dump(exclude_unset=True, mode="json"),
        )
        response.raise_for_status()
        return response.json()

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

    async def select_next_for_dive_frame_clustering(self) -> int | None:
        """Stage 1 cohort selector. See `select_next_for_laser_preprocessing`."""
        return await self._select_next("dive-frame-clustering")

    async def select_next_for_laser_preprocessing(self) -> int | None:
        """Stage 0.1 cohort selector: returns the next HIGH-priority
        dive needing laser preprocessing, or None when the cohort is
        empty. Server-side single-query equivalent of the api-worker's
        `select_next_high_priority_dive_for_laser_preprocessing_activity`.
        """
        return await self._select_next("laser-preprocessing")

    async def select_next_for_species_preprocessing(self) -> int | None:
        """Stage 2 cohort selector. See `select_next_for_laser_preprocessing`."""
        return await self._select_next("species-preprocessing")

    async def select_next_for_headtail_preprocessing(self) -> int | None:
        """Stage 5.1 cohort selector. See `select_next_for_laser_preprocessing`."""
        return await self._select_next("headtail-preprocessing")

    async def select_next_for_slate_preprocessing(self) -> int | None:
        """Stage 9 cohort selector. See `select_next_for_laser_preprocessing`."""
        return await self._select_next("slate-preprocessing")

    async def select_next_for_laser_prediction(self) -> int | None:
        """Laser-detector cohort selector. See `select_next_for_laser_preprocessing`."""
        return await self._select_next("laser-prediction")

    async def select_next_for_slate_prediction(self) -> int | None:
        """Slate-detector cohort selector. See `select_next_for_laser_preprocessing`."""
        return await self._select_next("slate-prediction")

    async def select_next_for_laser_calibration(self) -> int | None:
        """Stage 13 cohort selector. See `select_next_for_laser_preprocessing`."""
        return await self._select_next("laser-calibration")

    async def select_next_for_measure_fish(self) -> int | None:
        """Stage 14 cohort selector. See `select_next_for_laser_preprocessing`."""
        return await self._select_next("measure-fish")

    async def select_next_for_laser_depth(self) -> int | None:
        """Laser-depth cohort selector. See `select_next_for_laser_preprocessing`.

        Broader than stage 14's: any dive with a resolvable calibration and a
        validated laser label whose recorded depth is missing or was computed
        from a label or calibration that has since changed.
        """
        return await self._select_next("laser-depth")

    async def get_dives_needing_laser_population(self) -> list[int]:
        """Every dive needing model-assisted laser LS tasks (re)populated —
        prediction-gated. Returns all matches so the scheduled populate parent
        fans out one populate child per dive. See the
        `needing-laser-population` endpoint docstring."""
        response = await self._get("/api/v1/dives/needing-laser-population/")
        response.raise_for_status()
        return response.json() or []

    async def get_dives_needing_species_population(self) -> list[int]:
        """Every dive needing species LS tasks (re)populated onto a live
        project — superseded-aware, returns *all* matches (not one), so
        the scheduled populate parent can fan out one populate child per
        dive. See the `needing-species-population` endpoint docstring."""
        response = await self._get("/api/v1/dives/needing-species-population/")
        response.raise_for_status()
        return response.json() or []

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

    async def get_dive_laser_line(self, dive_id: int) -> DiveLaserLine | None:
        """Get the fitted laser-line fingerprint for a dive.

        Args:
            dive_id (int): The ID of the dive.

        Returns:
            DiveLaserLine | None: The dive's laser-line fingerprint, or None if
            it has not been fitted yet.
        """
        response = await self._get(f"/api/v1/dives/{dive_id}/laser-line/")
        if response.status_code == 404:
            return None
        response.raise_for_status()

        json = response.json()
        if json is None:
            return None
        return DiveLaserLine.model_validate(json)

    async def put_dive_laser_line(
        self, dive_id: int, laser_line: DiveLaserLine
    ) -> int:
        """Upsert the laser-line fingerprint for a dive.

        Args:
            dive_id (int): The ID of the dive.
            laser_line (DiveLaserLine): The fitted line + metrics to persist.

        Returns:
            int: The persisted row ID.
        """
        response = await self._put(
            f"/api/v1/dives/{dive_id}/laser-line/",
            json=laser_line.model_dump(exclude_unset=True, mode="json"),
        )
        response.raise_for_status()

        return response.json()

    async def get_calibration_candidates(
        self,
        dive_id: int,
        *,
        max_angle_deg: float | None = None,
        max_offset_px: float | None = None,
        min_confidence: float | None = None,
    ) -> List[CalibrationCandidate]:
        """Get ranked calibration-borrow candidates for a dive.

        Dives whose laser-line fingerprint matches this dive's (same camera,
        own extrinsics, confident fits, line within tolerance), ranked by line
        closeness. Suggest-only — pick one and call `set_calibration_source`.

        Args:
            dive_id (int): The dive to find borrow candidates for.
            max_angle_deg / max_offset_px / min_confidence: optional tolerance
            overrides; the server applies its defaults when omitted.

        Returns:
            List[CalibrationCandidate]: ranked candidates (may be empty).
        """
        query = {
            k: v
            for k, v in (
                ("max_angle_deg", max_angle_deg),
                ("max_offset_px", max_offset_px),
                ("min_confidence", min_confidence),
            )
            if v is not None
        }
        endpoint = f"/api/v1/dives/{dive_id}/calibration-candidates/"
        if query:
            endpoint = f"{endpoint}?{urlencode(query)}"
        response = await self._get(endpoint)
        response.raise_for_status()
        return [CalibrationCandidate.model_validate(x) for x in response.json()]

    async def set_calibration_source(
        self, dive_id: int, source_dive_id: int
    ) -> int:
        """Link a dive to borrow another dive's laser calibration.

        Use for a fish-only dive with no slate frames of its own — point
        it at a sibling slate/calibration dive shot with the same
        camera+laser rig. Laser-extrinsics resolution then falls back to
        the source dive when this dive has no calibration of its own.

        Args:
            dive_id (int): The dive that will borrow calibration.
            source_dive_id (int): The dive whose calibration to borrow.

        Returns:
            int: The linked dive's id.
        """
        response = await self._put(
            f"/api/v1/dives/{dive_id}/calibration-source/{source_dive_id}"
        )
        response.raise_for_status()

        return response.json()

    async def clear_calibration_source(self, dive_id: int) -> None:
        """Unlink a dive from any borrowed calibration source (idempotent).

        Args:
            dive_id (int): The dive to unlink.
        """
        response = await self._delete(
            f"/api/v1/dives/{dive_id}/calibration-source/"
        )
        response.raise_for_status()

    async def set_dive_slate(self, dive_id: int, dive_slate_id: int) -> int:
        """Set which DiveSlate template a dive was shot with.

        Identifies the physical slate (H-Slate / V-Slate N / Tic-Tac-Toe N),
        which the slate preprocess / sync / calibration stages need before a
        dive can be calibrated.

        Args:
            dive_id (int): The dive to set.
            dive_slate_id (int): The DiveSlate template id.

        Returns:
            int: The dive id.
        """
        response = await self._put(
            f"/api/v1/dives/{dive_id}/dive-slate/{dive_slate_id}"
        )
        response.raise_for_status()

        return response.json()
