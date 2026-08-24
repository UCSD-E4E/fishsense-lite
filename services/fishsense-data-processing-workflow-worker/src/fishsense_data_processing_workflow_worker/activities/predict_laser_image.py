"""Model-assisted laser labeling (data-worker, GPU).

Runs the fishsense-core `LaserDetector` (v2.2.0+) on one image and returns
the predicted laser dot in rectified-image pixels — the same space labelers
place `LaserLabel.x/y` — so the prediction can seed the laser Label Studio
task as a pre-annotation.

`torch` (via `fishsense_core[laser-detector]`) and the detector checkpoint are
only needed at run time, so both `fishsense_core.laser` and
`fishsense_core.image.linear_raw_image` are imported lazily inside the CPU/GPU
helper. That keeps this module importable without the extra installed and lets
the unit tests mock the detector.
"""

import asyncio
import logging
import os
import threading
from typing import Any

from temporalio import activity

from fishsense_data_processing_workflow_worker.object_store import (
    open_object_store_client,
)

_log = logging.getLogger(__name__)

# Checkpoint baked into the image (see the Dockerfile). Overridable via env for
# local runs / a future checkpoint swap.
DEFAULT_CHECKPOINT_PATH = os.environ.get(
    "E4EFS_LASER_DETECTOR__CHECKPOINT", "/e4efs/models/run3_epoch_021.pt"
)

# Module-level cache: the detector loads its weights once per worker process
# (loading is expensive and every per-image activity in the fan-out reuses it).
#
# The lock is load-bearing, not defensive. Activities run in a real
# ThreadPoolExecutor (`worker.py`'s `activity_executor` +
# `max_concurrent_activities`), so on a cold pod the whole first batch of
# per-image activities enters this together. Unguarded, each thread sees
# `_DETECTOR is None` and loads its own copy of the checkpoint onto the GPU.
_DETECTOR: Any = None
_DETECTOR_LOCK = threading.Lock()


def _load_detector(checkpoint_path: str) -> Any:
    """Build a LaserDetector from a local checkpoint. Imported lazily so the
    torch/segmentation-models extra is only required at run time."""
    # no-name-in-module: LaserDetector ships in fishsense-core >= 2.2.0; the
    # pinned wheel is bumped to 2.3.0 in the deps/GPU phase of this feature.
    from fishsense_core.laser import (  # pylint: disable=import-error,no-name-in-module
        LaserDetector,
    )

    return LaserDetector.from_checkpoint(checkpoint_path)


def _get_detector(checkpoint_path: str = DEFAULT_CHECKPOINT_PATH) -> Any:
    """Return the process-wide detector, loading it on first use.

    Double-checked locking: the fast path is a bare read for the common case
    (already loaded), and only the cold path pays for the lock. Safe under
    CPython because `_DETECTOR` is published by a single atomic assignment —
    a thread that sees a non-None value sees a fully-constructed detector.
    """
    global _DETECTOR  # pylint: disable=global-statement
    if _DETECTOR is not None:
        return _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            _log.info("loading LaserDetector checkpoint=%s", checkpoint_path)
            _DETECTOR = _load_detector(checkpoint_path)
    return _DETECTOR


def _predict_from_raw(
    raw_bytes: bytes,
    camera_matrix: list[list[float]],
    distortion_coefficients: list[float],
    wavelength: str | None,
    checkpoint_path: str,
) -> Any:
    """Off-loop CPU/GPU work: decode the raw bytes to a `LinearRawImage`
    (linear + Bayer-excess, which the 6-channel model needs) and run the
    detector with rectified output so the point lands in labeling space.

    Returns `(prediction, width, height)` — the fishsense-core
    `LaserPrediction` (`.x`, `.y`, `.confidence`) plus the rectified frame
    dimensions the x/y are relative to (needed to convert to Label Studio
    keypoint percentages downstream). `cv2.undistort` preserves the image
    size, so the rectified dims equal the decoded raw dims.
    """
    import numpy as np
    # no-name-in-module: linear_raw_image ships in fishsense-core >= 2.2.0
    # (bumped to 2.3.0 in the deps/GPU phase).
    from fishsense_core.image.linear_raw_image import (  # pylint: disable=import-error,no-name-in-module
        LinearRawImage,
    )

    image = LinearRawImage(raw_bytes)
    height, width = image.data.shape[:2]
    detector = _get_detector(checkpoint_path)
    prediction = detector.predict(
        image,
        wavelength=wavelength,
        rectify_output=True,
        camera_matrix=np.array(camera_matrix, dtype=float),
        distortion=np.array(distortion_coefficients, dtype=float),
    )
    return prediction, int(width), int(height)


def _models():
    from fishsense_data_processing_workflow_worker.workflows.predict_laser_images_workflow import (
        LaserPredictionResult,
        PredictLaserImageInput,
    )

    return PredictLaserImageInput, LaserPredictionResult


@activity.defn
async def predict_laser_image(payload):  # type: ignore[no-untyped-def]
    """Download one raw image from Garage, run the laser detector, and return
    its predicted laser dot keyed to `image_id`."""
    payload_cls, result_cls = _models()
    if not isinstance(payload, payload_cls):
        payload = payload_cls.model_validate(payload)

    activity.logger.info(
        "predicting laser image checksum=%s image_id=%d",
        payload.checksum,
        payload.image_id,
    )

    client = open_object_store_client()
    raw_bytes = await client.download_raw(payload.checksum)

    prediction, width, height = await asyncio.to_thread(
        _predict_from_raw,
        raw_bytes,
        payload.camera_matrix,
        payload.distortion_coefficients,
        payload.wavelength,
        DEFAULT_CHECKPOINT_PATH,
    )

    activity.logger.info(
        "predicted laser image_id=%d x=%s y=%s confidence=%.3f dims=%dx%d",
        payload.image_id,
        prediction.x,
        prediction.y,
        prediction.confidence,
        width,
        height,
    )
    return result_cls(
        image_id=payload.image_id,
        x=prediction.x,
        y=prediction.y,
        confidence=prediction.confidence,
        width=width,
        height=height,
    )
