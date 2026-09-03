"""Activity to resolve the per-image inputs the head/tail predict stage needs.

Returns a fully-populated `PredictHeadtailImagesInput` for the data-worker's
`PredictHeadtailImagesWorkflow`.

Lighter than the other predict resolvers: no camera intrinsics and no raw
bytes, because the stage reads the stage-5.1 JPEG that already exists in
Garage. What it does carry is the image's laser dots, which are not merely a
filter here — they are the crop centre the predictor looks around.

The image filter mirrors `select_next_for_headtail_prediction` exactly.
CLAUDE.md is blunt about why that matters: a resolver that disagrees with its
selector dispatches work the cohort never promised, and the dive can then never
drain — the parent re-fires on it every hour forever.
"""

from __future__ import annotations

from typing import List

from fishsense_shared.headtail_predictor import HEADTAIL_PREDICTOR_VERSION
from fishsense_shared.object_store import HEADTAIL_JPEG_FOLDER
from fishsense_shared.preprocess_contracts import (
    PredictHeadtailImage,
    PredictHeadtailImagesInput,
)
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


def select_images_needing_prediction(
    images, lasers, headtail_labels, predictions
) -> List[PredictHeadtailImage]:
    """Images whose prediction is missing or stale, with their live laser dots.

    An image qualifies when it has at least one *valid* laser dot, no completed
    human `HeadTailLabel`, and either no prediction or a stale one.

    "Stale" has two forms, collapsed here into one set because both need the
    detector run again and `put_head_tail_prediction` upserts on `image_id`:

    * a `predictor_version` other than the current one, NULL included — every
      row written before versioning carries NULL, and those are exactly the
      rows a bump exists to revisit; or
    * a `laser_label_id` that is no longer among the image's live labels.
      `get_laser_labels` filters superseded server-side, so a prediction naming
      a label absent from that set was made from a dot since dead-lettered, and
      the mask may be of the wrong thing entirely.

    "Labelled" keys off `completed`, NOT `label_studio_project_id`: populate
    seeds sentinel rows that carry a project id, so a project-id check would
    exclude every populate-seeded-but-unlabelled image and starve the detector
    on exactly the dives it should assist. That is the dive-84 case from the
    laser side.
    """
    labelled_ids = {
        label.image_id
        for label in headtail_labels
        if getattr(label, "completed", False)
    }

    dots_by_image: dict[int, list] = {}
    for laser in lasers:
        if not getattr(laser, "completed", False):
            continue
        if laser.x is None or laser.y is None:
            continue
        dots_by_image.setdefault(laser.image_id, []).append(laser)

    fresh_ids = set()
    for prediction in predictions:
        if getattr(prediction, "predictor_version", None) != HEADTAIL_PREDICTOR_VERSION:
            continue
        label_id = getattr(prediction, "laser_label_id", None)
        if label_id is not None:
            live_ids = {dot.id for dot in dots_by_image.get(prediction.image_id, [])}
            if label_id not in live_ids:
                continue  # the dot that chose the fish is gone
        fresh_ids.add(prediction.image_id)

    out: List[PredictHeadtailImage] = []
    for image in images:
        if not image.is_canonical:
            continue
        if image.id in labelled_ids or image.id in fresh_ids:
            continue
        dots = dots_by_image.get(image.id)
        if not dots:
            continue
        out.append(
            PredictHeadtailImage(
                image_id=image.id,
                checksum=image.checksum,
                laser_points=[[float(d.x), float(d.y)] for d in dots],
                laser_label_ids=[int(d.id) for d in dots],
            )
        )
    return out


@activity.defn
async def resolve_headtail_predict_inputs_activity(
    dive_id: int,
) -> PredictHeadtailImagesInput:
    activity.logger.info("resolving headtail predict inputs dive_id=%d", dive_id)
    async with get_fs_client() as fs:
        images = await fs.images.get(dive_id=dive_id) or []
        lasers = await fs.labels.get_laser_labels(dive_id) or []
        headtail_labels = await fs.labels.get_headtail_labels(dive_id) or []
        predictions = await fs.labels.get_headtail_predictions(dive_id) or []

        needing = select_images_needing_prediction(
            images, lasers, headtail_labels, predictions
        )
        activity.logger.info(
            "resolved headtail predict inputs dive_id=%d images=%d needing=%d",
            dive_id,
            len(images),
            len(needing),
        )
        return PredictHeadtailImagesInput(
            dive_id=dive_id,
            images=needing,
            jpeg_folder=HEADTAIL_JPEG_FOLDER,
        )
