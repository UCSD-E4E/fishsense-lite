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
    HEADTAIL_FALLBACK_PREDICTOR_VERSION,
    HEADTAIL_PREDICTOR_VERSION,
)
from fishsense_shared.preprocess_contracts import (
    HEADTAIL_STATUS_NO_UPGRADE_AVAILABLE,
    HeadtailPredictionResult,
)
from temporalio import activity
from temporalio.exceptions import ApplicationError

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
_FALLBACK_SEGMENTER: Any = None
_FALLBACK_LOCK = threading.Lock()

#: Re-exported: the status this activity emits for an image it cannot improve
#: on. Defined in `preprocess_contracts` because the api-worker parent has to
#: recognise it too -- see the note there.
#:
#: Not an abstention. An abstention is a statement about the *image* ("no fish
#: under the dot") and overwrites the row. This is a statement about the
#: *worker*, and overwriting would replace a fallback row with an identical
#: fallback row every hour for as long as the GPU is gone.
STATUS_NO_UPGRADE_AVAILABLE = HEADTAIL_STATUS_NO_UPGRADE_AVAILABLE


def cuda_available() -> bool:
    """Whether this process has a usable CUDA device.

    Wrapped rather than inlined because it decides which backend runs, and
    because a broken CUDA runtime should read as "no GPU" rather than escape
    as an import error. Same shape as `predict_slate_image._preferred_device`.
    """
    try:
        import torch  # pylint: disable=import-outside-toplevel,import-error

        return bool(torch.cuda.is_available())
    except Exception:  # pylint: disable=broad-except
        _log.debug("no usable CUDA device; head/tail predict will use fallback")
        return False


def _load_segmenter(checkpoint_path: str) -> Any:
    """Build the SAM3 concept segmenter. Imported lazily so torch/sam3 are only
    required at run time.

    **Loading prints four missing keys, and they are benign.** Expect:

        missing_keys=['backbone.vision_backbone.convs.3.conv_1x1.weight',
                      ...conv_1x1.bias, ...conv_3x3.weight, ...conv_3x3.bias']

    `Sam3DualViTDetNeck` builds one conv per entry in
    `scale_factors=(4.0, 2.0, 1.0, 0.5)` and runs all four, but
    `SAM3VLBackbone.forward` then does `sam3_features[:-scalp]` with
    `scalp=1`, discarding the lowest-resolution level -- which is exactly
    `convs.3`. Upstream trained with the same slice, so that conv was never
    trained and is not in the checkpoint. Its output cannot reach the
    encoder.

    Verified, not inferred: replacing all four `convs.3` tensors with
    `N(0, 50)` garbage leaves the returned masks bit-identical (8 masks,
    areas unchanged to the pixel).

    This matters because `_load_checkpoint` uses `strict=False` and only
    *prints* -- a genuinely wrong checkpoint would load just as quietly. So
    these four are the known-good baseline: if the set ever differs, that is
    the signal, and it is worth investigating rather than dismissing.
    """
    # pylint: disable=import-outside-toplevel,import-error
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if not cuda_available():
        # Fail fast and *non-retryably*. SAM 3.1 cannot be built without a GPU
        # at all: `build_sam3_image_model` reaches `PositionEmbeddingSine`,
        # which precomputes its cache with `torch.zeros(..., device="cuda")` --
        # hardcoded, no availability check -- so construction raises
        # `RuntimeError: No CUDA GPUs are available` before any of our device
        # handling is consulted.
        #
        # Retrying cannot help: the pod will not grow a GPU. Left retryable it
        # loops until the workflow times out while holding the pod, which is
        # what happened to all 356 activities of dive 94 on 2026-09-08.
        #
        # Reaching here at all is a bug now, not an environment: the caller
        # picks the fallback backend when there is no GPU. It stays as a guard
        # because "the pod lost its card" and "we chose wrong" both land here,
        # and both want to stop rather than spin.
        raise ApplicationError(
            "SAM 3.1 requires a GPU: build_sam3_image_model allocates its "
            "position-encoding cache on a hardcoded device='cuda'. This "
            "worker has no usable CUDA device.",
            type="NoGpuForSam3",
            non_retryable=True,
        )

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
    # Which tier produced this row. `HEADTAIL_PREDICTOR_VERSION` for SAM 3.1,
    # `HEADTAIL_FALLBACK_PREDICTOR_VERSION` for the Mask R-CNN fallback -- and
    # it is a parameter rather than the constant because the backend is chosen
    # at run time from whether this worker has a GPU. Stamping the fallback
    # value is what puts the row in the upgrade queue.
    predictor_version: int = HEADTAIL_PREDICTOR_VERSION


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
            predictor_version=options.predictor_version,
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
        "predictor_version": options.predictor_version,
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


def _to_numpy(mask) -> np.ndarray:
    """Mask -> ndarray, whatever the backend handed back.

    SAM3 returns torch tensors, and on the GPU worker they live on the device.
    `np.asarray` on a CUDA tensor raises rather than converting, so the whole
    stage would fail on the machine it is meant to run on while passing every
    test that stubs the backend. Detach first: the tensors carry grad history
    we neither need nor want to keep alive.
    """
    to_cpu = getattr(mask, "detach", None)
    if to_cpu is not None:
        mask = mask.detach().cpu()
    return np.asarray(mask)


def get_fallback_segmenter() -> Any:
    """Process-wide `FishSegmentation`, loaded on first use.

    Same double-checked locking as `get_segmenter`, for the same reason. No
    checkpoint argument: `fishsense_core.fish` ships in the base wheel with
    ONNX Runtime statically linked and the weights embedded in the `.so`,
    which is exactly why it can serve a GPU-less pod.
    """
    global _FALLBACK_SEGMENTER  # pylint: disable=global-statement
    if _FALLBACK_SEGMENTER is not None:
        return _FALLBACK_SEGMENTER
    with _FALLBACK_LOCK:
        if _FALLBACK_SEGMENTER is None:
            # pylint: disable=import-outside-toplevel,import-error,no-name-in-module
            from fishsense_core.fish import FishSegmentation

            _log.info("loading fishsense-core FishSegmentation (fallback backend)")
            segmentation = FishSegmentation()
            # `load_model()` is not optional and not lazy: without it
            # `inference` raises `ValueError: model has not been loaded`, which
            # is retryable and has no ceiling -- so the fallback would loop
            # until the child's 6h timeout, which is the exact failure this
            # backend exists to remove.
            #
            # Published only after loading, inside the lock, so a concurrent
            # caller can never see a constructed-but-unloaded segmenter.
            segmentation.load_model()
            _FALLBACK_SEGMENTER = segmentation
    return _FALLBACK_SEGMENTER


class _FishialAdapter:
    """Adapts `fishsense_core.fish.FishSegmentation` to the same seam.

    The fallback backend, used when this worker has no GPU. Three things
    differ from `_Sam3Adapter`, and each of them fails *silently* if got
    wrong, which is why they are asserted rather than assumed:

    * **BGR, not RGB.** `inference` expects the frame as `RectifiedImage.data`
      hands it over. Converting to RGB measurably degrades it and raises
      nothing.
    * **Landscape only.** `FishSegmentation` returns an empty mask, with no
      error, whenever width <= height. The 1800x1350 crop is landscape, so
      this holds today -- but it holds by arithmetic, not by contract, so it
      is checked rather than trusted.
    * **An instance label map, not per-instance masks.** `inference` returns
      one `(H, W)` array where each fish is a distinct non-zero integer, so
      it is split here to fit a seam whose contract is a list of binary masks.
    """

    def __init__(self, segmentation: Any):
        self._segmentation = segmentation

    def segment(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        height, width = image_bgr.shape[:2]
        if width <= height:
            _log.warning(
                "fallback segmenter given a %dx%d (non-landscape) crop; "
                "FishSegmentation returns an empty mask for these",
                width,
                height,
            )
            return []

        labels = np.asarray(self._segmentation.inference(image_bgr))
        ids = [int(i) for i in np.unique(labels) if int(i) != 0]
        return [(labels == i) for i in ids]


class _Sam3Adapter:
    """Adapts the SAM3 processor to the `segment(image) -> masks` seam."""

    def __init__(self, processor: Any, prompt: str = "fish"):
        self._processor = processor
        self._prompt = prompt

    def segment(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        """Run one concept-prompted segmentation, under autocast.

        The autocast context is required, not an optimisation. SAM 3.1's
        weights are bfloat16 and `Sam3Processor` sets up no autocast of its
        own, so without it fp32 activations meet bf16 weights and every frame
        raises `mat1 and mat2 must have the same dtype, but got BFloat16 and
        Float` inside `vitdet.forward`. It is a plain retryable
        `RuntimeError`, so in prod it looped on all 356 images of dive 94
        holding a GPU. Every upstream example enters the same context before
        inference.

        `device_type` is resolved per call rather than pinned to "cuda"
        because this stage's queue is served by either the GPU deployment or
        the CPU-fallback one, and `torch.autocast("cuda", ...)` on a CPU-only
        pod is a no-op that warns -- which would put the fallback straight
        back into the dtype mismatch this exists to prevent.

        Entered as a context manager, not `__enter__()` as the notebooks do:
        activities share a `ThreadPoolExecutor`, so leaking autocast would
        change the dtype regime of whatever ran next on this thread.
        """
        # pylint: disable=import-outside-toplevel
        import cv2
        import PIL.Image
        import torch

        # `set_image` returns the state, `set_text_prompt` *is* the inference
        # call and needs that state back, and the result is a dict rather than
        # an object. There is no `predict()`. Verified against a live model:
        # 8 masks at 0.78-0.95 confidence in 0.45 s warm.
        #
        # PIL, not the raw BGR array, and that is load-bearing rather than
        # tidiness: `set_image` does `height, width = image.shape[-2:]`, which
        # for an HWC numpy frame reads (1800, 3) -- and those numbers are what
        # every mask is finally interpolated to. Handing it the array yields
        # three-pixel-wide masks, silently and with no error.
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type, dtype=torch.bfloat16):
            state = self._processor.set_image(PIL.Image.fromarray(rgb))
            state = self._processor.set_text_prompt(self._prompt, state)
        masks = state.get("masks") if hasattr(state, "get") else None
        if masks is None:
            return []
        # (N, 1, H, W) bool; `squeeze` drops the channel axis. Bool survives
        # autocast untouched -- it is produced by a `> 0.5` comparison -- so
        # `_to_numpy` never sees a bfloat16 tensor numpy could not represent.
        return [_to_numpy(m).squeeze() for m in masks]


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

    client = open_object_store_client()
    on_gpu = cuda_available()

    if (
        not on_gpu
        and payload.has_existing_prediction
        and not (payload.existing_laser_superseded)
    ):
        # A GPU-less worker leaves every existing row alone. Rewriting a
        # fallback row produces an identical one -- the treadmill the upgrade
        # queue would otherwise cause every hour, since a fallback row is
        # permanently stale by design. Rewriting a SAM 3.1 row would be worse
        # still: a *downgrade*, reachable just by bumping
        # `HEADTAIL_PREDICTOR_VERSION` while no GPU is available.
        #
        # The exception is a superseded laser: the row may be of the wrong
        # fish entirely, and re-running even this backend fixes that.
        activity.logger.info(
            "skipping image_id=%d: already predicted and no GPU to improve on it",
            payload.image_id,
        )
        return HeadtailPredictionResult(
            image_id=payload.image_id,
            status=STATUS_NO_UPGRADE_AVAILABLE,
            predictor_version=HEADTAIL_FALLBACK_PREDICTOR_VERSION,
        )

    if on_gpu:
        sam3_cfg = _settings().sam3
        checkpoint = await ensure_checkpoint(
            client,
            sam3_cfg.cache_dir,
            sam3_cfg.model_name,
            sam3_cfg.model_version,
            sam3_cfg.checkpoint_filename,
        )
        # Off the loop for the same reason the download is: loading a multi-GB
        # checkpoint onto the GPU takes seconds, and this worker serves other
        # activities while it happens.
        segmenter = _Sam3Adapter(
            await asyncio.to_thread(get_segmenter, str(checkpoint))
        )
        options = PredictOptions(
            checkpoint=str(checkpoint),
            predictor_version=HEADTAIL_PREDICTOR_VERSION,
        )
    else:
        # No GPU: SAM 3.1 cannot even be constructed here (see
        # `_load_segmenter`), so run the backend that can. Worse than SAM 3.1,
        # far better than the empty task a labeler would otherwise get -- and
        # it is the backend the stage's original validation was measured on.
        segmenter = _FishialAdapter(await asyncio.to_thread(get_fallback_segmenter))
        options = PredictOptions(
            checkpoint="fishsense_core.fish.FishSegmentation",
            predictor_version=HEADTAIL_FALLBACK_PREDICTOR_VERSION,
        )

    jpeg = await client.download_processed_jpeg(payload.jpeg_folder, payload.checksum)

    result = await asyncio.to_thread(
        predict_from_jpeg,
        jpeg,
        payload.laser_points,
        segmenter,
        payload.image_id,
        payload.laser_label_ids,
        options,
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
