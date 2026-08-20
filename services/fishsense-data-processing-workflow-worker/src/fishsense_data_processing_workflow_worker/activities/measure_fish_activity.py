"""Activity to compute fish-length measurements for a dive (stage 14).

Ports `scripts/stage14_measure_fish.ipynb`. The depth-from-laser and
head/tail back-projection both delegate to
`fishsense_core.world_point.WorldPointHandler`; the math layer is
covered by synthetic-geometry tests in
`tests/test_compute_world_point_from_depth_convention.py` and
`tests/test_stage14_pipeline_sign_consistency.py`.

Lives on the data-processing worker for the same reason as stage 13 —
it pulls in fishsense-core math kernels; the api-worker stays thin.

Upstream dependency (real fish only): clusters with
`data_source=LABEL_STUDIO` must exist for the dive (stage 6.1). A
real-fish species label whose image isn't in any cluster is skipped
with a warning. Physical fish models (`content_of_image = "Fish Model,
<name>"`) are exempt — they carry no grouping labels and thus no
cluster, their identity is the model name (not the cluster), and the
length math uses only laser/head-tail/calibration; so the cluster gate
is waived for them.
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
from fishsense_shared import taxonomy
from temporalio import activity

from fishsense_data_processing_workflow_worker.activities.utils import (
    get_fs_client,
    load_dive_calibration_context,
)
from fishsense_data_processing_workflow_worker.laser_geometry import (
    compute_laser_point,
    measure_length_at_depth,
)


@dataclass
class MeasureFishResult:
    # pylint: disable=too-many-instance-attributes
    #   A counter bag, not a domain object: each field is one independent
    #   reason an image did or didn't get measured, and collapsing any of them
    #   together is what made debugging harder before they were split out.
    """Per-dive measurement summary.

    Surfaces what the notebook silently dropped: NaN-length results
    (laser/head/tail collinearity, missing depth) and species labels
    whose image wasn't reachable through laser/headtail/cluster lookups.
    """

    measured: int
    dropped_nan: int
    missing_laser_or_headtail: int
    missing_cluster: int
    skipped_already_measured: int = 0
    # Species rows whose `content_of_image` carries no `Common (Scientific)`
    # name — the non-`Fish` taxonomy branches (`Fish Model, Weasly Fish`,
    # `Calibration Targets, Ruler`). Counted separately because these were
    # previously tallied as `missing_laser_or_headtail`, which pointed
    # debugging at the labels instead of at the taxonomy branch.
    skipped_unmeasurable_species: int = 0
    # Measurements deleted because their Fish binding no longer matches the
    # image's species label — a model relabel (e.g. Snook -> Grouper) leaves
    # the row bound to the old model's Fish, and `post_measurement` upserts on
    # (image_id, fish_id) so re-measuring would ADD a row rather than replace.
    invalidated_stale_binding: int = 0
    # Images that already had a measurement, but one computed against a
    # calibration the dive no longer resolves to (or, for every row written
    # before `Measurement.laser_extrinsics_id` existed, against an unrecorded
    # one). Recomputed rather than skipped — the length is a function of the
    # extrinsics behind its depth, so replacing the calibration invalidates
    # it. Counted separately from `measured` because a run that is entirely
    # re-measures is the backfill draining, not new work arriving.
    remeasured_stale_calibration: int = 0


__all__ = ["MeasureFishResult", "measure_fish_activity"]


# The `content_of_image` taxonomy vocabulary — markers and parsers — lives in
# `fishsense_shared.taxonomy`, imported above. It used to be spelled here and
# again in the api's cohort selector and `dive_pipeline_status` view, kept in
# step by comments that had already drifted: both api copies documented
# "Fish Model, ..." and "Calibration Targets, Ruler" as *skipped* while their
# code matched both as measurable. One definition now, cross-checked against
# the SQL by `test_dive_pipeline_status_view.py` over
# `taxonomy.MEASURABILITY_CORPUS`.
_parse_model_name = taxonomy.parse_model_name
_parse_species_names = taxonomy.parse_species_names


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


async def _ensure_model_fish(
    fs: Client,
    name: str,
    cache: dict[str, Fish],
) -> Fish:
    """Find-or-create the Fish for a physical model, keyed GLOBALLY by name.

    Unlike `_ensure_fish`, identity is the model name, not the cluster: the same
    model resolves to one Fish across dives/cameras/time, and two models in one
    cluster resolve to two Fish. `cluster.fish_id` is never touched. `cache`
    dedups within a single dive run; `get_by_name` dedups across runs; and
    `post` upserts on name, so a concurrent create still converges to one row.
    """
    if name in cache:
        return cache[name]
    fish = await fs.fish.get_by_name(name)
    if fish is None:
        fish = Fish(id=None, name=name, species_id=None)
        fish.id = await fs.fish.post(fish)
    cache[name] = fish
    return fish


def _measure_length(
    laser_label: LaserLabel,
    headtail_label: HeadTailLabel,
    laser_extrinsics: LaserExtrinsics,
    camera_intrinsics: CameraIntrinsics,
) -> float:
    """Triangulate fish length in meters from a single (laser, headtail)
    observation. Returns NaN when the geometry is degenerate (the handler
    surfaces this rather than raising).

    The projection itself lives in `laser_geometry`, shared with
    `compute_laser_depths_activity` — the depth this derives on the way to a
    length is the same number that stage records per image, and two
    transcriptions of a kernel whose sign convention already needed a
    synthetic-geometry investigation is one too many.
    """
    point = compute_laser_point(laser_label, laser_extrinsics, camera_intrinsics)
    return measure_length_at_depth(headtail_label, point.depth_m, camera_intrinsics)


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


async def _fetch_measurements_by_image(fs, dive_id: int) -> dict[int, list]:
    """Existing measurements for this dive, grouped by image id.

    One fetch per dive, not per image — the caller loops over every
    top-three label. `None` is the SDK's "no measurements yet" signal.
    Grouped (rather than a bare id set) so the caller can also check WHICH
    Fish each row is bound to and invalidate stale bindings.
    """
    existing = await fs.fish.get_measurements(dive_id) or []
    by_image: dict[int, list] = {}
    for measurement in existing:
        if measurement.image_id is not None:
            by_image.setdefault(measurement.image_id, []).append(measurement)
    return by_image


def _has_complete_keypoints(laser_label, headtail_label) -> bool:
    """True iff both labels exist and every keypoint coord is set."""
    if laser_label is None or headtail_label is None:
        return False
    return (
        laser_label.x is not None
        and laser_label.y is not None
        and headtail_label.head_x is not None
        and headtail_label.head_y is not None
        and headtail_label.tail_x is not None
        and headtail_label.tail_y is not None
    )


@activity.defn
async def measure_fish_activity(dive_id: int) -> MeasureFishResult:
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    # Orchestration function — gathers dive/camera/laser/cluster/label
    # context plus per-iteration locals. Splitting it would just push
    # the same state into a parameter list of a helper. The same applies
    # to the statement count: the body is a flat sequence of guard →
    # log → count → continue per skip reason, which reads better inline
    # than dispersed across helpers.
    """Walk the dive's top-three species labels and write a `Measurement`
    for each one whose laser + headtail + cluster context is present and
    whose triangulated length is finite.

    Raises `ValueError` for missing prerequisites that should fail loud:
    the dive itself, its camera intrinsics, or its `laser_extrinsics`
    (run stage 13 first).
    """
    async with get_fs_client() as fs:
        context = await load_dive_calibration_context(fs, dive_id)
        camera_intrinsics = context.camera_intrinsics
        laser_extrinsics = context.laser_extrinsics

        species_labels = await fs.labels.get_species_labels(dive_id) or []
        clusters = (
            await fs.images.get_clusters(dive_id, DataSource.LABEL_STUDIO.value) or []
        )
        cluster_by_image = _index_clusters_by_image(clusters)
        top_three = _filter_top_three(species_labels)

        # `post_measurement` upserts on (image_id, fish_id) so a duplicate
        # can't be recorded, but re-measuring still means re-deriving a
        # length and re-binding a fish for work already done — skip it.
        measurements_by_image = await _fetch_measurements_by_image(fs, dive_id)

        result = MeasureFishResult(
            measured=0,
            dropped_nan=0,
            missing_laser_or_headtail=0,
            missing_cluster=0,
            skipped_already_measured=0,
        )
        # Name -> Fish for physical models, deduped within this dive run.
        model_fish_cache: dict[str, Fish] = {}

        for species_label in top_three:
            image_id = species_label.image_id

            # Classify the taxonomy branch first — the stale-binding check
            # below needs to know which Fish this image *should* map to.
            # Real fish carry a "Common (Scientific)" name; models carry a
            # "Fish Model, <name>" prefix. Anything else (Calibration
            # Targets, empty) is not measurable.
            names = _parse_species_names(species_label.content_of_image)
            model_name = _parse_model_name(species_label.content_of_image)
            if names is None and model_name is None:
                activity.logger.info(
                    "dive_id=%d image_id=%d: content_of_image=%r is neither a "
                    "'Common (Scientific)' name nor a 'Fish Model,' row; not "
                    "measurable, skipping",
                    dive_id,
                    image_id,
                    species_label.content_of_image,
                )
                result.skipped_unmeasurable_species += 1
                continue

            existing_measurements = measurements_by_image.get(image_id, [])
            if model_name is not None and existing_measurements:
                # Model identity is the name, so the correct Fish is knowable
                # up front. Any row bound to a different Fish is stale — the
                # image was measured under a previous species label. Delete it
                # rather than re-measuring around it: `post_measurement`
                # upserts on (image_id, fish_id), so a corrected binding would
                # be ADDED alongside and the image counted twice.
                expected = await _ensure_model_fish(fs, model_name, model_fish_cache)
                stale = [m for m in existing_measurements if m.fish_id != expected.id]
                for measurement in stale:
                    activity.logger.info(
                        "dive_id=%d image_id=%d: measurement bound to fish_id=%s "
                        "but the label now says %r (fish_id=%s); invalidating "
                        "the stale binding",
                        dive_id,
                        image_id,
                        measurement.fish_id,
                        model_name,
                        expected.id,
                    )
                    await fs.fish.delete_measurement(measurement.fish_id, image_id)
                    result.invalidated_stale_binding += 1
                existing_measurements = [
                    m for m in existing_measurements if m.fish_id == expected.id
                ]

            # "Already measured" has to mean "already measured with the
            # calibration this dive resolves to now". A recalibration (the
            # 2026-08-11 panel-offset fix hit 6 of 8 measured dives) changes
            # the depth behind every length, and the unqualified skip left
            # those wrong lengths standing indefinitely. Re-measuring is safe
            # and cheap: `post_measurement` upserts on (image_id, fish_id),
            # so the stale row is replaced in place.
            measured_with_current_calibration = [
                m
                for m in existing_measurements
                if m.laser_extrinsics_id == laser_extrinsics.id
            ]
            if measured_with_current_calibration:
                activity.logger.info(
                    "dive_id=%d image_id=%d: already measured; skipping",
                    dive_id,
                    image_id,
                )
                result.skipped_already_measured += 1
                continue
            if existing_measurements:
                activity.logger.info(
                    "dive_id=%d image_id=%d: measured against laser_extrinsics_id=%s "
                    "but the dive now resolves to %s; re-measuring",
                    dive_id,
                    image_id,
                    [m.laser_extrinsics_id for m in existing_measurements],
                    laser_extrinsics.id,
                )
                result.remeasured_stale_calibration += 1

            # Real fish anchor identity to their LABEL_STUDIO cluster, so a
            # missing cluster is a hard skip. Models are keyed by name and use
            # no cluster (they carry none — no grouping labels), so the gate is
            # waived for them.
            cluster = cluster_by_image.get(image_id)
            if names is not None and cluster is None:
                activity.logger.warning(
                    "dive_id=%d image_id=%d: no LABEL_STUDIO cluster; skipping",
                    dive_id,
                    image_id,
                )
                result.missing_cluster += 1
                continue

            laser_label = await fs.labels.get_laser_label(image_id=image_id)
            headtail_label = await fs.labels.get_headtail_label(image_id=image_id)
            if not _has_complete_keypoints(laser_label, headtail_label):
                activity.logger.warning(
                    "dive_id=%d image_id=%d: missing laser/headtail; skipping",
                    dive_id,
                    image_id,
                )
                result.missing_laser_or_headtail += 1
                continue

            if names is not None:
                common, scientific = names
                species = await _ensure_species(fs, common, scientific)
                fish = await _ensure_fish(fs, cluster, species)
            else:
                fish = await _ensure_model_fish(fs, model_name, model_fish_cache)

            length_m = _measure_length(
                laser_label, headtail_label, laser_extrinsics, camera_intrinsics
            )
            if not np.isfinite(length_m):
                activity.logger.warning(
                    "dive_id=%d image_id=%d fish_id=%s: non-finite length=%s; "
                    "skipping",
                    dive_id,
                    image_id,
                    fish.id,
                    length_m,
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
                    # Which calibration this length came from. Without it the
                    # skip above cannot tell a current measurement from one
                    # computed against extrinsics that have since been
                    # replaced.
                    laser_extrinsics_id=laser_extrinsics.id,
                ),
            )
            result.measured += 1

            activity.heartbeat()

        activity.logger.info("dive_id=%d measure complete: %s", dive_id, result)
        return result
