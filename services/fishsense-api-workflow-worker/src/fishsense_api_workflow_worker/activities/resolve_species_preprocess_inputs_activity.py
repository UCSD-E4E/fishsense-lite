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

An eligible image in **no** PREDICTION cluster is emitted as its own
singleton cluster. Stage 1 clustering is one-shot per dive, so a frame it
missed can never be clustered afterwards; iterating clusters alone left
those frames without a stage-2 JPEG, which kept the whole species LS
project an unpublished draft (populate publishes only when nothing is
deferred). See the orphan block below for the measured prod impact.
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
    activity.logger.info("resolving species preprocess inputs dive_id=%d", dive_id)
    async with get_fs_client() as fs:
        dive = await fs.dives.get(dive_id=dive_id)
        if dive is None:
            raise ValueError(f"dive_id={dive_id} not found")
        if dive.camera_id is None:
            raise ValueError(f"dive_id={dive_id} has no camera_id")

        intrinsics = await fs.cameras.get_intrinsics(dive.camera_id)
        if intrinsics is None:
            raise ValueError(f"camera_id={dive.camera_id} has no intrinsics")

        prediction_clusters = (
            await fs.images.get_clusters(dive_id, DataSource.PREDICTION.value) or []
        )
        images = await fs.images.get(dive_id=dive_id) or []
        # Canonical frames only. The same physical frames live under several
        # dive rows (half of prod's image table is duplicate content), and
        # `is_canonical` marks which copy is the real one. The cohort selectors
        # gate on it, and CLAUDE.md requires resolvers to mirror the selector
        # predicate exactly -- otherwise the dispatched per-image work would not
        # match what the cohort promised, and the dive could never drain.
        # This also covers on-demand/backfill runs, which bypass the cohort.
        images = [image for image in images if image.is_canonical]
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
        clustered_image_ids: set[int] = set()
        for cluster in prediction_clusters:
            cluster_checksums = []
            for image_id in cluster.image_ids or []:
                clustered_image_ids.add(image_id)
                if (
                    image_id in checksum_by_id
                    and image_id in valid_laser_image_ids
                    and image_id not in labeled_image_ids
                ):
                    cluster_checksums.append(checksum_by_id[image_id])
            if cluster_checksums:
                clusters.append(cluster_checksums)

        # Orphans: eligible images that belong to NO PREDICTION cluster.
        #
        # Stage 1 clustering is one-shot per dive -- its cohort excludes any
        # dive that already has a PREDICTION cluster -- so a frame it missed
        # can never be clustered later. Reaching images only by iterating
        # clusters therefore stranded them permanently: no stage-2 JPEG, so
        # species populate's JPEG gate deferred them forever.
        #
        # That is not "one image missing". Populate publishes the LS project
        # only when `deferred == 0`, so a single orphan keeps the ENTIRE
        # project an unpublished draft, invisible to annotators. Prod
        # 2026-08-28: dive 442 had 6 orphans holding back 352 tasks; dive 87
        # had 18 of 405; dive 94 had 8 of 356. The rate is never zero.
        #
        # Each orphan is emitted as its own singleton so the per-image overlay
        # renders "image 1 of 1". Grouping them into one synthetic cluster
        # would be a lie about temporal grouping and would label unrelated
        # frames "image i of N".
        orphan_checksums = [
            checksum_by_id[image.id]
            for image in images
            if image.id not in clustered_image_ids
            and image.id in valid_laser_image_ids
            and image.id not in labeled_image_ids
        ]
        clusters.extend([checksum] for checksum in orphan_checksums)

        activity.logger.info(
            "resolved species preprocess inputs dive_id=%d "
            "prediction_clusters=%d non_empty_clusters=%d orphans=%d "
            "total_checksums=%d",
            dive_id,
            len(prediction_clusters),
            len(clusters) - len(orphan_checksums),
            len(orphan_checksums),
            sum(len(c) for c in clusters),
        )
        return PreprocessSpeciesImagesInput(
            dive_id=dive_id,
            clusters=clusters,
            camera_matrix=intrinsics.camera_matrix.tolist(),
            distortion_coefficients=intrinsics.distortion_coefficients.tolist(),
        )
