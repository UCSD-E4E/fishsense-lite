"""Model-assisted head/tail labeling (data-worker, GPU).

Predicts snout/fork keypoints for one image: a SAM3 mask of the fish the
image's validated laser dot sits on, keypointed by `fishsense_core`'s
`FishHeadTailDetector`, in the same rectified-frame pixels a labeler clicks in.
The head/tail populate step emits the result as a Label Studio pre-annotation.

Three design choices are measured rather than assumed
(docs/plans/headtail-prediction.md §0.2b, §0.2c):

* **Read the stage-5.1 JPEG, not the raw `.ORF`.** That JPEG is the exact frame
  the labeler is shown, so predicting on anything else would be predicting on a
  different image than the one being labelled. It also removes NAS staging and
  the rawpy decode entirely.
* **Crop, don't tile.** The laser dot already says where the fish is, so one
  1800x1350 window centred on it reaches the same resolution a ~20-tile sweep
  would, for a single inference — 0.6 s against 11.3 s — and scores slightly
  better, because the fish is centred and never split across a tile boundary.
* **SAM3, not the fishsense-core Mask R-CNN.** Measured on 80 frames against
  human labels: 35.0% usable predictions against 23.8%, at a fifth of the time.
  Note the gain is coverage, not per-mask precision — conditional on predicting
  at all, the two are within noise of each other.

`sam3` and `torch` are imported lazily inside the loader so this module stays
importable without them, which is what lets the unit tests drive the whole
pipeline with a stub segmenter.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np
from fishsense_shared.headtail_predictor import (
    HEADTAIL_CROP_HEIGHT,
    HEADTAIL_CROP_WIDTH,
    HEADTAIL_PREDICTOR_VERSION,
)
from fishsense_shared.preprocess_contracts import HeadtailPredictionResult
from temporalio import activity

from fishsense_data_processing_workflow_worker.headtail_geometry import (
    crop_origin,
    lift_point,
    mask_at_point,
    silhouette_ratio,
)

_log = logging.getLogger(__name__)

# Module-level cache: the weights load once per worker process. The lock is
# load-bearing, not defensive — activities run in a real ThreadPoolExecutor, so
# on a cold pod the whole first batch of per-image activities enters together
# and an unguarded load would give every thread its own copy of a
# multi-gigabyte model.
_SEGMENTER: Any = None
_SEGMENTER_LOCK = threading.Lock()


def _load_segmenter(checkpoint_path: str) -> Any:
    """Build the SAM3 concept segmenter. Imported lazily so torch/sam3 are only
    required at run time."""
    # pylint: disable=import-outside-toplevel,import-error
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(checkpoint_path=checkpoint_path)
    model.eval()
    return Sam3Processor(model)


def get_segmenter(checkpoint_path: str) -> Any:
    """Return the process-wide segmenter, loading it on first use.

    Double-checked locking: the fast path is a bare read for the common case
    (already loaded), and only the cold path pays for the lock. Safe under
    CPython because `_SEGMENTER` is published by a single atomic assignment — a
    thread that sees a non-None value sees a fully-constructed segmenter.
    """
    global _SEGMENTER  # pylint: disable=global-statement
    if _SEGMENTER is not None:
        return _SEGMENTER
    with _SEGMENTER_LOCK:
        if _SEGMENTER is None:
            _log.info("loading SAM3 checkpoint=%s", checkpoint_path)
            _SEGMENTER = _load_segmenter(checkpoint_path)
    return _SEGMENTER


@dataclass(frozen=True)
class PredictOptions:
    """Window size and provenance for one prediction.

    Grouped rather than passed loose because they travel together and none of
    them changes per image: the crop is a tuned constant, and the checkpoint and
    core version are recorded on every row without being decided on.
    """

    crop_w: int = HEADTAIL_CROP_WIDTH
    crop_h: int = HEADTAIL_CROP_HEIGHT
    checkpoint: Optional[str] = None
    core_version: Optional[str] = None


def _laser_label_for_mask(
    local_points: Sequence[Sequence[float]],
    laser_label_ids: Optional[Sequence[int]],
    binary,
) -> Optional[int]:
    """Which laser label landed on the chosen mask, if any were supplied."""
    if not laser_label_ids:
        return None
    for (px, py), label_id in zip(local_points, laser_label_ids):
        xi, yi = int(round(px)), int(round(py))
        if 0 <= yi < binary.shape[0] and 0 <= xi < binary.shape[1] and binary[yi, xi]:
            return int(label_id)
    return None


def predict_from_jpeg(
    jpeg_bytes: bytes,
    laser_points: Sequence[Sequence[float]],
    segmenter: Any,
    image_id: int,
    laser_label_ids: Optional[Sequence[int]] = None,
    options: Optional[PredictOptions] = None,
) -> HeadtailPredictionResult:
    """Decode, crop, segment, gate, keypoint, and lift back to frame pixels.

    `segmenter` is anything with `segment(image) -> list[np.ndarray]` returning
    crop-local binary masks; production passes the SAM3 adapter, tests pass a
    stub. Keeping the model behind that seam is what makes the crop and lift
    arithmetic — the parts that fail plausibly — testable without a GPU.
    """
    # pylint: disable=import-outside-toplevel
    import cv2

    options = options or PredictOptions()

    def _abstain(status: str, **extra) -> HeadtailPredictionResult:
        return HeadtailPredictionResult(
            image_id=image_id,
            status=status,
            predictor_version=HEADTAIL_PREDICTOR_VERSION,
            checkpoint=options.checkpoint,
            core_version=options.core_version,
            **extra,
        )

    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return _abstain("decode_failed")
    height, width = frame.shape[:2]

    if not laser_points:
        return _abstain("laser_off_all_fish", width=width, height=height)

    origin_x, origin_y = crop_origin(
        laser_points[0][0],
        laser_points[0][1],
        width,
        height,
        options.crop_w,
        options.crop_h,
    )
    crop = np.ascontiguousarray(
        frame[
            origin_y : origin_y + options.crop_h,
            origin_x : origin_x + options.crop_w,
        ]
    )

    masks = segmenter.segment(crop)
    if not masks:
        return _abstain(
            "no_detections",
            width=width,
            height=height,
            crop_x=origin_x,
            crop_y=origin_y,
        )

    # The gate, in crop coordinates.
    local_points = [(px - origin_x, py - origin_y) for px, py in laser_points]
    mask = mask_at_point(masks, local_points)
    if mask is None:
        return _abstain(
            "laser_off_all_fish",
            width=width,
            height=height,
            crop_x=origin_x,
            crop_y=origin_y,
        )

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    return _keypoint(
        binary,
        image_id=image_id,
        origin=(origin_x, origin_y),
        frame_size=(width, height),
        local_points=local_points,
        laser_label_ids=laser_label_ids,
        options=options,
    )


def _keypoint(
    binary,
    *,
    image_id: int,
    origin: tuple,
    frame_size: tuple,
    local_points: Sequence[Sequence[float]],
    laser_label_ids: Optional[Sequence[int]],
    options: "PredictOptions",
) -> HeadtailPredictionResult:
    """Keypoint one chosen mask and lift the result into frame coordinates.

    Split out of `predict_from_jpeg` so the decode/crop/gate half and the
    keypoint/lift half each stay readable; the lift is the step that fails
    plausibly, so it is worth being able to see all of it at once.
    """
    # pylint: disable=import-outside-toplevel
    from fishsense_core.fish import FishHeadTailDetector

    origin_x, origin_y = origin
    area = int(np.count_nonzero(binary))
    common = {
        "image_id": image_id,
        "width": frame_size[0],
        "height": frame_size[1],
        "crop_x": origin_x,
        "crop_y": origin_y,
        "predictor_version": HEADTAIL_PREDICTOR_VERSION,
        "checkpoint": options.checkpoint,
        "core_version": options.core_version,
    }

    try:
        head, tail = FishHeadTailDetector().find_head_tail_img(binary * 255)
    # The detector is a PyO3 native call whose error surface is not a
    # documented exception hierarchy, and one unfittable mask must not fail the
    # whole per-image activity.
    # pylint: disable-next=broad-exception-caught
    except Exception as exc:
        _log.warning("image_id=%d find_head_tail_img failed: %s", image_id, exc)
        return HeadtailPredictionResult(
            status="headtail_failed", mask_area_px=area, **common
        )

    head_x, head_y = lift_point(head, origin_x, origin_y)
    tail_x, tail_y = lift_point(tail, origin_x, origin_y)
    length = float(np.hypot(head_x - tail_x, head_y - tail_y))

    return HeadtailPredictionResult(
        status="predicted",
        head_x=head_x,
        head_y=head_y,
        tail_x=tail_x,
        tail_y=tail_y,
        mask_area_px=area,
        silhouette_ratio=silhouette_ratio(area, length),
        # Which laser label chose this fish, so a later supersede makes the row
        # selectable as stale rather than leaving it unnoticed.
        laser_label_id=_laser_label_for_mask(local_points, laser_label_ids, binary),
        **common,
    )


class _Sam3Adapter:
    """Adapts the SAM3 processor to the `segment(image) -> masks` seam."""

    def __init__(self, processor: Any, prompt: str = "fish"):
        self._processor = processor
        self._prompt = prompt

    def segment(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        # pylint: disable=import-outside-toplevel
        import cv2

        self._processor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        self._processor.set_text_prompt(self._prompt)
        output = self._processor.predict()
        masks = getattr(output, "masks", None)
        if masks is None:
            return []
        return [np.asarray(m).squeeze() for m in masks]


def _settings():
    # Function-local so importing this module doesn't trigger Dynaconf's eager
    # validation — see the config gotcha in CLAUDE.md.
    from fishsense_data_processing_workflow_worker.config import (  # pylint: disable=import-outside-toplevel
        settings,
    )

    return settings


@activity.defn
async def predict_headtail_image(payload):  # type: ignore[no-untyped-def]
    """Download one stage-5.1 JPEG, predict its head/tail, key it to image_id."""
    # pylint: disable=import-outside-toplevel
    from fishsense_shared.preprocess_contracts import PredictHeadtailImage

    from fishsense_data_processing_workflow_worker.checkpoint_cache import (
        ensure_checkpoint,
    )
    from fishsense_data_processing_workflow_worker.object_store import (
        open_object_store_client,
    )

    if not isinstance(payload, PredictHeadtailImage):
        payload = PredictHeadtailImage.model_validate(payload)

    sam3_cfg = _settings().sam3
    client = open_object_store_client()

    checkpoint = await ensure_checkpoint(
        client,
        sam3_cfg.cache_dir,
        sam3_cfg.model_name,
        sam3_cfg.model_version,
        sam3_cfg.checkpoint_filename,
    )
    segmenter = _Sam3Adapter(get_segmenter(str(checkpoint)))

    jpeg = await client.download_processed_jpeg(payload.jpeg_folder, payload.checksum)

    result = await asyncio.to_thread(
        predict_from_jpeg,
        jpeg,
        payload.laser_points,
        segmenter,
        payload.image_id,
        payload.laser_label_ids,
        PredictOptions(checkpoint=str(checkpoint)),
    )
    activity.logger.info(
        "predicted headtail image_id=%d status=%s crop=(%s,%s) ratio=%s",
        result.image_id,
        result.status,
        result.crop_x,
        result.crop_y,
        None if result.silhouette_ratio is None else round(result.silhouette_ratio, 3),
    )
    return result
