"""Activity to compute fish-length measurements for a dive (stage 14).

Ports `scripts/stage14_measure_fish.ipynb`. The depth-from-laser and
head/tail back-projection both delegate to
`fishsense_core.world_point.WorldPointHandler`; the math layer is
covered by synthetic-geometry tests in
`tests/test_compute_world_point_from_depth_convention.py` and
`tests/test_stage14_pipeline_sign_consistency.py`.

Lives on the data-processing worker for the same reason as stage 13 —
it pulls in fishsense-core math kernels; the api-worker stays thin.

Upstream dependency: clusters with `data_source=LABEL_STUDIO` must
exist for the dive (stage 6.1). Species labels whose image isn't in
any cluster are skipped with a warning — handles the partial-port
case gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from fishsense_api_sdk.client import Client
from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_api_sdk.models.data_source import DataSource
from fishsense_api_sdk.models.dive_frame_cluster import DiveFrameCluster
from fishsense_api_sdk.models.fish import Fish
from fishsense_api_sdk.models.headtail_label import HeadTailLabel
from fishsense_api_sdk.models.laser_extrinsics import LaserExtrinsics
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_sdk.models.measurement import Measurement
from fishsense_api_sdk.models.species import Species
from fishsense_api_sdk.models.species_label import SpeciesLabel
from fishsense_core.world_point import WorldPointHandler
from temporalio import activity

from fishsense_data_processing_workflow_worker.activities.utils import get_fs_client


@dataclass
class MeasureFishResult:
    """Per-dive measurement summary.

    Surfaces what the notebook silently dropped: NaN-length results
    (laser/head/tail collinearity, missing depth) and species labels
    whose image wasn't reachable through laser/headtail/cluster lookups.
    """

    measured: int
    dropped_nan: int
    missing_laser_or_headtail: int
    missing_cluster: int


__all__ = ["MeasureFishResult", "measure_fish_activity"]


def _parse_species_names(content_of_image: str | None) -> tuple[str, str] | None:
    """Pull (common_name, scientific_name) out of the species label's
    `content_of_image` field. Format: "..., Common Name (Scientific name)".

    Returns None if the field is empty or doesn't match the expected
    shape (we'd rather skip than write a malformed Species row).
    """
    if not content_of_image:
        return None
    last_chunk = content_of_image.split(", ")[-1]
    if "(" not in last_chunk or not last_chunk.endswith(")"):
        return None
    common = last_chunk.split(" (")[0].strip()
    scientific = last_chunk.split(" (")[-1][:-1].strip()
    if not common or not scientific:
        return None
    return common, scientific


async def _ensure_species(fs: Client, common: str, scientific: str) -> Species:
    """Idempotent find-or-create on (scientific_name)."""
    species = await fs.fish.get_species_by_scientific_name(scientific)
    if species is not None:
        return species
    new = Species(id=None, common_name=common, scientific_name=scientific)
    new.id = await fs.fish.post_species(new)
    return new


async def _ensure_fish(
    fs: Client,
    cluster: DiveFrameCluster,
    species: Species,
) -> Fish:
    """Find-or-create a Fish for the cluster and rebind the cluster
    (`put_cluster`) when its `fish_id` doesn't yet point at this fish."""
    fish = (
        await fs.fish.get(fish_id=cluster.fish_id)
        if cluster.fish_id is not None
        else None
    )
    if fish is None:
        fish = Fish(id=None, species_id=species.id)
        fish.id = await fs.fish.post(fish)

    if cluster.fish_id != fish.id:
        cluster.fish_id = fish.id
        await fs.images.put_cluster(cluster.dive_id, cluster.id, cluster)

    return fish


def _measure_length(
    laser_label: LaserLabel,
    headtail_label: HeadTailLabel,
    laser_extrinsics: LaserExtrinsics,
    camera_intrinsics: CameraIntrinsics,
) -> float:
    """Triangulate fish length in meters from a single (laser, headtail)
    observation. Returns NaN when the geometry is degenerate (handler
    surfaces this rather than raising)."""
    k_inv = np.linalg.inv(camera_intrinsics.camera_matrix)
    handler = WorldPointHandler(k_inv)

    laser2d = np.array([laser_label.x, laser_label.y])
    laser3d = handler.compute_world_point_from_laser(
        laser_extrinsics.laser_position,
        laser_extrinsics.laser_axis,
        laser2d,
    )
    depth = float(laser3d[2])

    head3d = handler.compute_world_point_from_depth(
        np.array([headtail_label.head_x, headtail_label.head_y]), depth
    )
    tail3d = handler.compute_world_point_from_depth(
        np.array([headtail_label.tail_x, headtail_label.tail_y]), depth
    )
    return float(np.linalg.norm(head3d - tail3d))


def _index_clusters_by_image(
    clusters: Iterable[DiveFrameCluster],
) -> dict[int, DiveFrameCluster]:
    by_image: dict[int, DiveFrameCluster] = {}
    for cluster in clusters:
        for image_id in cluster.image_ids:
            by_image[image_id] = cluster
    return by_image


def _filter_top_three(
    species_labels: Iterable[SpeciesLabel],
) -> list[SpeciesLabel]:
    return [
        label
        for label in species_labels
        if label.top_three_photos_of_group and label.image_id is not None
    ]


@activity.defn
async def measure_fish_activity(dive_id: int) -> MeasureFishResult:
    """Walk the dive's top-three species labels and write a `Measurement`
    for each one whose laser + headtail + cluster context is present and
    whose triangulated length is finite.

    Raises `ValueError` for missing prerequisites that should fail loud:
    the dive itself, its camera intrinsics, or its `laser_extrinsics`
    (run stage 13 first).
    """
    async with get_fs_client() as fs:
        dive = await fs.dives.get(dive_id=dive_id)
        if dive is None:
            raise ValueError(f"dive_id={dive_id} not found")
        if dive.camera_id is None:
            raise ValueError(f"dive_id={dive_id} has no camera_id")

        camera_intrinsics = await fs.cameras.get_intrinsics(dive.camera_id)
        if camera_intrinsics is None:
            raise ValueError(f"camera_id={dive.camera_id} has no intrinsics")

        laser_extrinsics = await fs.dives.get_laser_extrinsics(dive_id)
        if laser_extrinsics is None:
            raise ValueError(
                f"dive_id={dive_id} has no laser_extrinsics; "
                "run perform_laser_calibration_activity first"
            )

        species_labels = await fs.labels.get_species_labels(dive_id) or []
        clusters = (
            await fs.images.get_clusters(dive_id, DataSource.LABEL_STUDIO.value)
            or []
        )
        cluster_by_image = _index_clusters_by_image(clusters)
        top_three = _filter_top_three(species_labels)

        result = MeasureFishResult(
            measured=0,
            dropped_nan=0,
            missing_laser_or_headtail=0,
            missing_cluster=0,
        )

        for species_label in top_three:
            image_id = species_label.image_id

            cluster = cluster_by_image.get(image_id)
            if cluster is None:
                activity.logger.warning(
                    "dive_id=%d image_id=%d: no LABEL_STUDIO cluster; skipping",
                    dive_id, image_id,
                )
                result.missing_cluster += 1
                continue

            laser_label = await fs.labels.get_laser_label(image_id=image_id)
            headtail_label = await fs.labels.get_headtail_label(image_id=image_id)
            if (
                laser_label is None
                or laser_label.x is None
                or laser_label.y is None
                or headtail_label is None
                or headtail_label.head_x is None
                or headtail_label.head_y is None
                or headtail_label.tail_x is None
                or headtail_label.tail_y is None
            ):
                activity.logger.warning(
                    "dive_id=%d image_id=%d: missing laser/headtail; skipping",
                    dive_id, image_id,
                )
                result.missing_laser_or_headtail += 1
                continue

            names = _parse_species_names(species_label.content_of_image)
            if names is None:
                activity.logger.warning(
                    "dive_id=%d image_id=%d: unparseable content_of_image=%r; skipping",
                    dive_id, image_id, species_label.content_of_image,
                )
                result.missing_laser_or_headtail += 1
                continue
            common, scientific = names

            species = await _ensure_species(fs, common, scientific)
            fish = await _ensure_fish(fs, cluster, species)

            length_m = _measure_length(
                laser_label, headtail_label, laser_extrinsics, camera_intrinsics
            )
            if not np.isfinite(length_m):
                activity.logger.warning(
                    "dive_id=%d image_id=%d fish_id=%s: non-finite length=%s; "
                    "skipping",
                    dive_id, image_id, fish.id, length_m,
                )
                result.dropped_nan += 1
                continue

            await fs.fish.post_measurement(
                fish.id,
                Measurement(
                    id=None,
                    fish_id=fish.id,
                    image_id=image_id,
                    length_m=length_m,
                ),
            )
            result.measured += 1

            activity.heartbeat()

        activity.logger.info(
            "dive_id=%d measure complete: %s", dive_id, result
        )
        return result
