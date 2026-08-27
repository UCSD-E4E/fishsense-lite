"""Activity to resolve the per-image inputs stage 0.1 needs for a dive.

Returns a fully-populated `PreprocessLaserImagesInput` ready to hand
to the data-worker's child workflow. Image filter mirrors the API
selector's "image with no LaserLabel row at all" predicate — once
any row exists for an image (even an incomplete one seeded by
populate), the image's preprocessed JPEG is on the file-exchange and
shouldn't be regenerated. Without this matching filter the parent
selector would drop the dive but the resolver would still return
fresh per-image work for any partially-seeded dive — wasted CPU.

The bbox lives here rather than baked into the data-worker so the
api-worker can swap to a per-camera box once we ship a second sensor
without touching the data-worker. See `DEFAULT_LASER_BBOX` for how
its value was measured.
"""

from __future__ import annotations

from typing import List, Tuple

from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_shared import PreprocessLaserImagesInput
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client

# Subject distances the rig actually works at, padded either side of the
# observed range. `LaserDepth` over 1109 prod images spans 0.44-5.45 m
# (p01 0.54, p50 1.71, p99 3.50).
WORKING_DEPTH_RANGE_M: Tuple[float, float] = (0.35, 8.0)

# The laser is mounted ~3 cm left of and ~10 cm below the camera, so its dot
# does not sit in a fixed place: it traces an epipolar streak that runs up and
# to the left as the subject gets closer, converging on a per-rig asymptote as
# it gets further away. This region has to cover that streak for *every* rig,
# because stage 0.1 runs before the dive has a calibration of its own —
# calibration is stage 13.
#
# It is a polygon rather than a rectangle because the union of those streaks is
# genuinely not axis-aligned, and the corners a rectangle adds are exactly where
# the laser cannot be. At equal coverage the polygon is 26% smaller than the
# tightest rectangle (1.12 vs 1.52 Mpx) — worth having, since the whole point of
# the overlay is to tell a labeler where to look.
#
# Measured 2026-08-27 against prod (13 `LaserExtrinsics` rows; 31,322 completed
# non-superseded `LaserLabel` rows across 262 dives): the convex hull of the
# projected calibration rays plus every dive's observed laser locus, dilated by
# 150 px and simplified to 8 vertices. It holds every calibrated ray, all 190
# well-populated dives' loci, and 99.90% of the labels — each of the ~30
# stragglers being a specular-reflection mislabel that survived the RANSAC
# supersede pass, not a laser. `(2217, 2088)` recurs verbatim across dives
# 246/375/446, which is a fixed artifact of the rig rather than anything anyone
# shot.
#
# The predecessor, [1800, 700, 2400, 1600], was the original notebook constant
# and was wrong in a way no single dive would reveal: across the corpus it
# clipped the left edge (103 labels), the bottom (76) and the top (33), and two
# dives had their *median* laser outside it altogether — for those it pointed
# labelers away from the laser rather than at it. Its right edge was the one it
# had about right, which is why it survived: it held 99.35% of all labels and
# looks fine on most dives.
#
# Deriving it from the 13 calibrations alone would still have clipped: only 13
# of the 262 labelled dives have ever been calibrated, and the uncalibrated
# rigs sit measurably further right — dives 253/397/455/468 carry median laser
# x at 2213..2267, past the 3-sigma envelope of the calibrated set. Both legs
# are pinned in `tests/test_default_laser_region.py`; re-measure there rather
# than nudging these numbers.
LASER_REGION_POLYGON: List[List[int]] = [
    [1580, 570],
    [1700, 465],
    [2335, 395],
    [2455, 525],
    [2470, 1610],
    [2185, 1890],
    [1920, 1905],
    [1625, 1365],
]

# The polygon's bounding box, and only ever that — `test_default_laser_region`
# pins the two together. It exists because the api-worker and the data-worker
# deploy independently (in-slot converge vs. `kubectl apply` on NRP, often days
# apart), so a data-worker that predates the polygon has to keep drawing
# something correct. See `PreprocessLaserImagesInput.laser_region`.
DEFAULT_LASER_BBOX: List[int] = [1580, 395, 2470, 1905]


def _select_unlabeled_images(
    images: List[Image], existing_labels: List[LaserLabel]
) -> List[Image]:
    """Return images needing a stage-0.1 JPEG: no non-sentinel LaserLabel
    row, or one flagged `needs_reprocess`.

    "Non-sentinel" = `label_studio_project_id is not None`. NULL-
    project rows are legacy sentinels (see API selector docstring).
    Once populate seeds a real row for an image, the image's
    preprocessed JPEG is on the file-exchange and the image drops
    out of the preprocess work set.

    `needs_reprocess` puts it back: it is how a change to what the overlay
    draws reaches an image whose JPEG was already written. Mirrors
    `select_next_for_laser_preprocessing` exactly, which is not optional --
    a selector that picks a dive this function then finds no work for would
    stage the dive's raw `.ORF`s from the NAS every hour forever.
    """
    labeled_image_ids = {
        label.image_id
        for label in existing_labels
        if label.label_studio_project_id is not None
    }
    flagged_image_ids = {
        label.image_id for label in existing_labels if label.needs_reprocess
    }
    return [
        image
        for image in images
        if image.id not in labeled_image_ids or image.id in flagged_image_ids
    ]


@activity.defn
async def resolve_laser_preprocess_inputs_activity(
    dive_id: int,
) -> PreprocessLaserImagesInput:
    activity.logger.info("resolving laser preprocess inputs dive_id=%d", dive_id)
    async with get_fs_client() as fs:
        dive = await fs.dives.get(dive_id=dive_id)
        if dive is None:
            raise ValueError(f"dive_id={dive_id} not found")
        if dive.camera_id is None:
            raise ValueError(f"dive_id={dive_id} has no camera_id")

        intrinsics = await fs.cameras.get_intrinsics(dive.camera_id)
        if intrinsics is None:
            raise ValueError(f"camera_id={dive.camera_id} has no intrinsics")

        images = await fs.images.get(dive_id=dive_id) or []

        # Canonical frames only. The same physical frames live under several

        # dive rows (half of prod's image table is duplicate content), and

        # `is_canonical` marks which copy is the real one. The cohort selectors

        # gate on it, and CLAUDE.md requires resolvers to mirror the selector

        # predicate exactly -- otherwise the dispatched per-image work would not

        # match what the cohort promised, and the dive could never drain.

        # This also covers on-demand/backfill runs, which bypass the cohort.

        images = [image for image in images if image.is_canonical]
        existing_labels = await fs.labels.get_laser_labels(dive_id) or []
        unlabeled = _select_unlabeled_images(images, existing_labels)

        activity.logger.info(
            "resolved laser preprocess inputs dive_id=%d images=%d unlabeled=%d",
            dive_id,
            len(images),
            len(unlabeled),
        )
        return PreprocessLaserImagesInput(
            dive_id=dive_id,
            image_checksums=[image.checksum for image in unlabeled],
            camera_matrix=intrinsics.camera_matrix.tolist(),
            distortion_coefficients=intrinsics.distortion_coefficients.tolist(),
            bbox=list(DEFAULT_LASER_BBOX),
            laser_region=[list(vertex) for vertex in LASER_REGION_POLYGON],
        )
