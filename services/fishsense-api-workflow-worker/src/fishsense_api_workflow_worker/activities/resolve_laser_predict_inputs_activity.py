"""Activity to resolve the per-image inputs the laser-detector stage needs.

Returns a fully-populated `PredictLaserImagesInput` for the data-worker's
`PredictLaserImagesWorkflow`. Image filter mirrors the API selector: an image
needs a prediction only if it has no `LaserPrediction` and no non-sentinel
`LaserLabel` — so re-runs return only still-unpredicted images and the model
never re-predicts a labeled image.
"""

from __future__ import annotations

from typing import List

from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_sdk.models.laser_prediction import LaserPrediction
from fishsense_shared import (
    LASER_PREDICTOR_VERSION,
    LASER_REGION_POLYGON,
    PredictLaserImage,
    PredictLaserImagesInput,
)
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


def _select_images_needing_prediction(
    images: List[Image],
    predictions: List[LaserPrediction],
    labels: List[LaserLabel],
) -> List[Image]:
    """Images whose prediction is missing or stale, and that no human has
    finished labeling.

    "Stale" means a `predictor_version` other than the current one, NULL
    included -- every row written before versioning carries NULL, and those are
    exactly the rows a bump exists to revisit. Collapsing "no prediction" and
    "old prediction" into one set is deliberate: both need the detector run
    against them, and `put_laser_prediction` upserts on `image_id`, so a re-run
    overwrites rather than duplicating.

    Keys off `completed`, NOT `label_studio_project_id`: the laser populate
    step seeds placeholder rows (`completed=False`) that carry a project_id,
    so a project-id check would exclude every populate-seeded-but-unlabeled
    image and the detector would never predict it (the dive-84 / project-274728
    case). `labels` is already superseded-filtered by `get_laser_labels`, so a
    completed row here is a live human label. Mirrors the API selector
    (`select_next_for_laser_prediction`) and populate's own definition.

    The selector's dive-level "still being labeled" gate is deliberately *not*
    mirrored here. That gate decides which dives are worth spending GPU time
    on; once a dive is selected, every stale image in it is work. Re-checking
    it per image could only produce disagreement with the selector, which is
    the failure this file's neighbours keep warning about.
    """
    labeled_ids = {label.image_id for label in labels if label.completed}
    current_ids = {
        prediction.image_id
        for prediction in predictions
        if getattr(prediction, "predictor_version", None) == LASER_PREDICTOR_VERSION
    }
    return [
        image
        for image in images
        if image.id not in current_ids and image.id not in labeled_ids
    ]


@activity.defn
async def resolve_laser_predict_inputs_activity(
    dive_id: int,
) -> PredictLaserImagesInput:
    activity.logger.info("resolving laser predict inputs dive_id=%d", dive_id)
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
        predictions = await fs.labels.get_laser_predictions(dive_id) or []
        labels = await fs.labels.get_laser_labels(dive_id) or []
        needing = _select_images_needing_prediction(images, predictions, labels)

        activity.logger.info(
            "resolved laser predict inputs dive_id=%d images=%d needing=%d",
            dive_id,
            len(images),
            len(needing),
        )
        return PredictLaserImagesInput(
            dive_id=dive_id,
            images=[
                PredictLaserImage(image_id=image.id, checksum=image.checksum)
                for image in needing
            ],
            camera_matrix=intrinsics.camera_matrix.tolist(),
            distortion_coefficients=intrinsics.distortion_coefficients.tolist(),
            # Laser color isn't tracked per-dive yet; the model uses its
            # "unknown wavelength" channel. Wire a real value here if/when a
            # per-dive laser color lands.
            wavelength=None,
            # Predictions outside this are dropped by the data-worker. See
            # `fishsense_shared.laser_region` for how it was measured.
            laser_region=[list(vertex) for vertex in LASER_REGION_POLYGON],
        )
