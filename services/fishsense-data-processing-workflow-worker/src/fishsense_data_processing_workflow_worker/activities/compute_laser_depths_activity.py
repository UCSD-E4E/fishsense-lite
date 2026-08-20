"""Activity to record how far away each laser dot was, for a whole dive.

Stage 14 has always derived this number — it projects the laser dot, takes the
depth, and back-projects head and tail against it — and then thrown it away,
so the distance to the laser existed only for the frames stage 14 measures and
was never queryable afterwards. This activity computes it for **every** image
in a dive whose laser label was validated, and stores it as `LaserDepth`.

Lives on the data-worker for the same reason as stages 13 and 14: the
projection kernel is `fishsense_core.world_point.WorldPointHandler`, and no
other service depends on fishsense-core. Unlike the preprocess stages there is
no image data involved at all — no NAS staging, no object store, no per-image
fan-out — it is pure SDK math over labels the pipeline already holds.

Each row also carries the triangulation's `residual_m` — how close the camera
ray and the laser ray actually came to meeting, ~0 when the dot is genuinely
consistent with the calibration. Recorded rather than acted on: it is blind to
error along the laser's epipolar line and it is metric, so a threshold has to
come from the distribution this stage is the first thing to produce. The gate
that rejects work stays `depth_m > 0`.

Idempotent and self-healing. Each row records the laser label and the
calibration it came from, and an image is recomputed exactly when either has
moved on. That makes the first sweep over a dive its own backfill and every
subsequent one a cheap no-op, without a separate one-shot script.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fishsense_api_sdk.models.laser_depth import LaserDepth
from temporalio import activity

from fishsense_data_processing_workflow_worker.activities.utils import get_fs_client
from fishsense_data_processing_workflow_worker.laser_geometry import compute_laser_point

__all__ = ["ComputeLaserDepthsResult", "compute_laser_depths_activity"]


@dataclass
class ComputeLaserDepthsResult:
    """Per-dive summary, in the shape of the other stage results.

    The skip counters are the interesting half: a dive that is entirely
    `skipped_current` is settled, while a non-zero `skipped_invalid_geometry`
    names labels whose position cannot be reconciled with the dive's
    calibration at all.
    """

    computed: int = 0
    # Already recorded against this label and this calibration.
    skipped_current: int = 0
    # Not a validated fix: incomplete, superseded, or missing a coordinate.
    skipped_unusable_label: int = 0
    # The laser ray and the camera ray do not meet in front of the camera.
    skipped_invalid_geometry: int = 0


def _is_validated_fix(laser_label) -> bool:
    """The repo-wide *valid laser* gate, as the cohort selector spells it in
    SQL: the labeler placed a point, the validator signed off, and the
    per-dive RANSAC fit hasn't superseded it."""
    return bool(
        laser_label.completed
        and not laser_label.superseded
        and laser_label.x is not None
        and laser_label.y is not None
    )


@activity.defn
async def compute_laser_depths_activity(dive_id: int) -> ComputeLaserDepthsResult:
    """Store the distance to the laser dot for each of the dive's validated
    laser labels.

    Raises `ValueError` for missing prerequisites that should fail loud rather
    than silently produce nothing: the dive, its camera intrinsics, or its
    resolved `laser_extrinsics` (run stage 13, or link a calibration source,
    first). The cohort selector only offers dives that have all three, so
    reaching this raise means the selector and this activity disagree — the
    failure mode that leaves a dive in the cohort forever.
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

        # Resolves own-then-borrowed via `Dive.calibration_dive_id`, so a
        # fish-only dive gets its sibling's rig calibration transparently.
        laser_extrinsics = await fs.dives.get_laser_extrinsics(dive_id)
        if laser_extrinsics is None:
            raise ValueError(
                f"dive_id={dive_id} has no laser_extrinsics; "
                "run perform_laser_calibration_activity first"
            )

        laser_labels = await fs.labels.get_laser_labels(dive_id) or []
        # One dive-scoped read rather than a lookup per image — same shape
        # stage 14 uses for its already-measured filter.
        existing_by_image = {
            depth.image_id: depth for depth in await fs.images.get_laser_depths(dive_id)
        }

        result = ComputeLaserDepthsResult()

        for laser_label in laser_labels:
            image_id = laser_label.image_id
            if image_id is None or not _is_validated_fix(laser_label):
                result.skipped_unusable_label += 1
                continue

            existing = existing_by_image.get(image_id)
            if (
                existing is not None
                and existing.laser_label_id == laser_label.id
                and existing.laser_extrinsics_id == laser_extrinsics.id
            ):
                result.skipped_current += 1
                continue

            point = compute_laser_point(
                laser_label, laser_extrinsics, camera_intrinsics
            )
            # `depth_m > 0`, not `isfinite`. The kernel reports "these rays
            # never meet" as the origin and a dot on the wrong side of the
            # principal point as a negative Z — both finite, and a length
            # computed at a negative depth comes out identical to its
            # positive twin, so nothing downstream would notice. See
            # `test_laser_geometry.py`.
            if not math.isfinite(point.depth_m) or point.depth_m <= 0.0:
                activity.logger.warning(
                    "dive_id=%d image_id=%d laser_label_id=%s: laser at (%s, %s) "
                    "gives depth=%s (residual=%s) against "
                    "laser_extrinsics_id=%s; no valid intersection, skipping",
                    dive_id,
                    image_id,
                    laser_label.id,
                    laser_label.x,
                    laser_label.y,
                    point.depth_m,
                    point.residual_m,
                    laser_extrinsics.id,
                )
                result.skipped_invalid_geometry += 1
                continue

            await fs.images.put_laser_depth(
                image_id,
                LaserDepth(
                    id=None,
                    depth_m=point.depth_m,
                    range_m=point.range_m,
                    # How well the dot and the calibration agreed. Recorded,
                    # not gated on: it cannot see error along the laser's
                    # epipolar line, and being metric it means different
                    # things at 0.9 m and 2.5 m — so a threshold belongs to
                    # the observed distribution, which this is the first
                    # thing to produce.
                    residual_m=point.residual_m,
                    image_id=image_id,
                    laser_label_id=laser_label.id,
                    laser_extrinsics_id=laser_extrinsics.id,
                ),
            )
            result.computed += 1

            activity.heartbeat()

        activity.logger.info("dive_id=%d laser depths complete: %s", dive_id, result)
        return result
