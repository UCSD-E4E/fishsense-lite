"""Activity to resolve the per-cluster inputs stage 2 needs for a dive.

Returns a fully-populated `PreprocessSpeciesImagesInput` ready to hand
to the data-worker's child workflow. Clusters preserve the temporal
grouping from `DiveFrameCluster(data_source=PREDICTION)` so the
per-image overlay can render "image i of N" for each cluster.

Cluster image_ids are filtered at resolver granularity to images
that:
  1. Carry a *valid* LaserLabel (completed, not superseded, both x/y
     populated) — same gate the API cohort selector uses.
  2. Have no non-sentinel SpeciesLabel row (`project_id IS NOT NULL`)
     — so a re-firing on the same cohort dive doesn't re-import LS
     tasks for already-populated images.

Empty clusters that survive nothing through the filter are dropped
so the data-worker fan-out doesn't waste an empty slot.
"""

from __future__ import annotations

from typing import List

from fishsense_api_sdk.models.data_source import DataSource
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_shared import PreprocessSpeciesImagesInput
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


def _is_valid_laser(label: LaserLabel) -> bool:
    """Same predicate the API SQL uses for the cohort gate."""
    return bool(
        label.completed
        and not label.superseded
        and label.x is not None
        and label.y is not None
    )


@activity.defn
async def resolve_species_preprocess_inputs_activity(
    dive_id: int,
) -> PreprocessSpeciesImagesInput:
    async with get_fs_client() as fs:
        dive = await fs.dives.get(dive_id=dive_id)
        if dive is None:
            raise ValueError(f"dive_id={dive_id} not found")
        if dive.camera_id is None:
            raise ValueError(f"dive_id={dive_id} has no camera_id")

        intrinsics = await fs.cameras.get_intrinsics(dive.camera_id)
        if intrinsics is None:
            raise ValueError(
                f"camera_id={dive.camera_id} has no intrinsics"
            )

        prediction_clusters = (
            await fs.images.get_clusters(dive_id, DataSource.PREDICTION.value)
            or []
        )
        images = await fs.images.get(dive_id=dive_id) or []
        checksum_by_id = {image.id: image.checksum for image in images}

        laser_labels = await fs.labels.get_laser_labels(dive_id) or []
        valid_laser_image_ids = {
            label.image_id for label in laser_labels if _is_valid_laser(label)
        }

        existing_species = await fs.labels.get_species_labels(dive_id) or []
        # Sentinel-aware: drop images that have at least one non-sentinel
        # species row (real project_id), matching the API selector.
        labeled_image_ids = {
            label.image_id
            for label in existing_species
            if label.label_studio_project_id is not None
        }

        clusters: List[List[str]] = []
        for cluster in prediction_clusters:
            cluster_checksums = [
                checksum_by_id[image_id]
                for image_id in (cluster.image_ids or [])
                if image_id in checksum_by_id
                and image_id in valid_laser_image_ids
                and image_id not in labeled_image_ids
            ]
            if cluster_checksums:
                clusters.append(cluster_checksums)

        return PreprocessSpeciesImagesInput(
            dive_id=dive_id,
            clusters=clusters,
            camera_matrix=intrinsics.camera_matrix.tolist(),
            distortion_coefficients=intrinsics.distortion_coefficients.tolist(),
        )
