"""Model-assisted slate labeling (data-worker, CPU).

**RETIRED 2026-08-03 — registered, but nothing schedules it.** The
ECC >= 0.80 acceptance gate does not transfer out of distribution: pool
dives produced high-ECC (0.93-0.97) *false* fits that sailed through it
(prod dives 65/71/77/80/83, all pool). The team declined an
active-learning loop; `predict-slate-images-workflow-schedule` is now
actively deleted at worker startup (`worker._RETIRED_SCHEDULE_IDS`) and
the 130 seeded Label Studio predictions were removed.

The code is kept registered so a future evaluation can start it by hand
— it is dormant, not dead — but nothing invokes it on its own. Do not
read it as part of the live pipeline.


Runs the fishsense-core dive-slate estimator (`fishsense_core.slate`, v2.4.0+)
on one rectified frame and returns the board's reference points in
rectified-photo pixels — the same space the sync activity stores
`DiveSlateLabel.reference_points` (after stripping the composite panel offset) —
so the prediction can seed the dive-slate Label Studio task as a pre-annotation.

The estimator is CPU-only (~200 ms/frame). This first cut uses the classical
path (`board_mask=None`), a supported estimate_plane mode that costs ~13 points
of coverage vs the learned `BoardMasker` mask; the mask (behind the `[slate]`
extra, a HuggingFace checkpoint) is a follow-on.

Gating mirrors `slate_training.predict.predict_slate`: only V-Slate / Tic-Tac-Toe
families, ECC >= the confidence floor, and every projected point on the photo.
A bad pre-annotation is worse than none (a labeler may accept it into the
corpus), so every rejection returns a *reason* and the default is to decline.
"""

import asyncio
import logging
import os
import threading
from typing import Any, Sequence

from temporalio import activity

from fishsense_data_processing_workflow_worker.object_store import (
    open_object_store_client,
)

_log = logging.getLogger(__name__)

# Optional learned board mask (BoardMasker, behind fishsense_core[slate]). A
# local checkpoint path overrides the HuggingFace download. The classical path
# (board_mask=None) is a supported fallback (~13 pts less coverage), so any
# load/inference failure degrades gracefully instead of failing the frame.
DEFAULT_SLATE_CHECKPOINT_PATH = os.environ.get("E4EFS_SLATE_DETECTOR__CHECKPOINT", "")
_MASKER: Any = None
_MASKER_LOADED = False
# See `_get_masker` — the flag must only ever be published *after* `_MASKER`,
# and the lock is what stops a concurrent caller reading the pair mid-load.
_MASKER_LOCK = threading.Lock()


def _get_masker() -> Any:
    """Return the process-wide BoardMasker, or None if it can't be loaded.

    Caches the outcome (including failure) so a missing mask doesn't retry the
    load on every frame. Loads from a local checkpoint when
    `E4EFS_SLATE_DETECTOR__CHECKPOINT` points at one, else from HuggingFace.

    `_MASKER_LOADED` is set in a `finally`, *after* `_MASKER` is assigned, and
    the whole cold path is serialized. Setting the flag first was a real bug:
    `from_pretrained()` reaches HuggingFace and takes seconds, and activities
    run in a real thread pool, so every frame arriving during that window read
    the flag as True, got a still-unset `_MASKER`, and silently fell back to
    the classical path. Silently, because nothing raised — the degradation
    warning below only fires for an actual load failure.
    """
    global _MASKER, _MASKER_LOADED  # pylint: disable=global-statement
    if _MASKER_LOADED:
        return _MASKER
    with _MASKER_LOCK:
        if _MASKER_LOADED:
            return _MASKER
        try:
            # no-name-in-module: BoardMasker ships in fishsense_core[slate] >= 2.4.0
            from fishsense_core.slate import (  # pylint: disable=import-error,no-name-in-module
                BoardMasker,
            )

            if DEFAULT_SLATE_CHECKPOINT_PATH and os.path.exists(
                DEFAULT_SLATE_CHECKPOINT_PATH
            ):
                _MASKER = BoardMasker.from_checkpoint(DEFAULT_SLATE_CHECKPOINT_PATH)
            else:
                _MASKER = BoardMasker.from_pretrained()
            _log.info("loaded BoardMasker (learned board mask)")
        except Exception as exc:  # pylint: disable=broad-except
            # torch/hf missing, no network, or bad checkpoint — fall back to the
            # classical estimator. The mask only adds coverage; it's never required.
            _log.warning(
                "BoardMasker unavailable (%s); using classical slate path", exc
            )
            _MASKER = None
        finally:
            _MASKER_LOADED = True
    return _MASKER


# ECC >= 0.80 keeps ~58% of frames at median ~6 px (measured on the corpus in
# slate_training/docs/design.md). Coverage matters more than the last few px for
# assisted labeling. Only meaningful within the current estimator config.
DEFAULT_MIN_CONFIDENCE = 0.80

# V-Slate and Tic-Tac-Toe are measured-good; H-Slate is excluded on zero
# evidence (no labeled frames), not bad evidence — enable it when one exists.
SUPPORTED_FAMILIES = frozenset({"v-slate", "tic-tac-toe"})


def slate_family(slate_name: str) -> str:
    """Coarse family key for a `DiveSlate.name` ('V-Slate 3' -> 'v-slate')."""
    name = (slate_name or "").strip().lower()
    if name.startswith("v-slate"):
        return "v-slate"
    if name.startswith("tic-tac-toe"):
        return "tic-tac-toe"
    if name.startswith("h-slate"):
        return "h-slate"
    return name


def gate_estimate(
    estimate: Any,
    slate_name: str,
    width: int,
    height: int,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[list[list[float]] | None, float, str | None]:
    """Decide whether an estimate is seedable. Pure — no OpenCV/model work.

    Returns `(reference_points | None, confidence, rejected_reason)`. Points are
    the estimate's projected image points, in rectified-photo pixels, only when
    all gates pass; otherwise None + a reason. Mirrors `predict_slate`'s posture:
    decline unless every gate passes.
    """
    if slate_family(slate_name) not in SUPPORTED_FAMILIES:
        return None, 0.0, "unsupported_slate_family"
    if estimate is None:
        return None, 0.0, "no_board"

    confidence = float(estimate.ecc_score)
    if confidence < min_confidence:
        return None, confidence, "low_confidence"

    points = [[float(x), float(y)] for x, y in estimate.image_points]
    # A partial set breaks stage-13's positional pairing — reject if any point
    # falls off the photo (the photo-space analog of predict_slate's
    # points_off_canvas gate; the composite check happens in populate).
    if any(not (0.0 <= x <= width and 0.0 <= y <= height) for x, y in points):
        return None, confidence, "points_off_canvas"

    return points, confidence, None


def _render_template_gray(pdf_bytes: bytes, dpi: int):
    """Render page 0 of the slate template PDF to a grayscale array at `dpi`
    (the DiveSlate.dpi the reference points are expressed in, so template pixels
    and template_points share a scale)."""
    import numpy as np
    import pymupdf

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        page = document.load_page(0)
        pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width
    )


def _predict_from_bytes(  # pylint: disable=too-many-locals
    raw_bytes: bytes,
    pdf_bytes: bytes,
    camera_matrix: list[list[float]],
    distortion_coefficients: list[float],
    dpi: float,
    template_points: Sequence[Sequence[float]],
    slate_name: str,
    min_confidence: float,
) -> tuple[list[list[float]] | None, float, str | None, int, int]:
    """Off-loop CPU work: rectify the raw frame, render the template, run the
    fishsense-core board-plane estimator, and gate the result.

    Returns `(reference_points | None, confidence, reason, width, height)`.
    """
    import numpy as np
    from fishsense_api_sdk.models.camera_intrinsics import (
        CameraIntrinsics,
    )
    from fishsense_core.image.raw_image import (  # pylint: disable=no-name-in-module
        RawImage,
    )
    from fishsense_core.image.rectified_image import (  # pylint: disable=no-name-in-module
        RectifiedImage,
    )
    from fishsense_core.slate import (  # pylint: disable=no-name-in-module
        estimate_plane,
    )

    cam = np.array(camera_matrix, dtype=float)
    intrinsics = CameraIntrinsics(
        camera_matrix=cam,
        distortion_coefficients=np.array(distortion_coefficients, dtype=float),
        camera_id=None,
    )
    bgr = RectifiedImage(RawImage(raw_bytes), intrinsics).data
    height, width = bgr.shape[:2]

    template_gray = _render_template_gray(pdf_bytes, dpi=int(dpi))

    # Learned board mask improves localization when available; None is the
    # supported classical fallback.
    board_mask = None
    masker = _get_masker()
    if masker is not None:
        try:
            board_mask = masker.predict(bgr)
        except Exception as exc:  # pylint: disable=broad-except
            _log.warning("board mask inference failed (%s); classical slate path", exc)

    estimate = estimate_plane(
        bgr,
        template_gray,
        [[float(x), float(y)] for x, y in template_points],
        float(dpi),
        cam,
        board_mask=board_mask,
    )
    points, confidence, reason = gate_estimate(
        estimate, slate_name, int(width), int(height), min_confidence
    )
    return points, confidence, reason, int(width), int(height)


def _models():
    from fishsense_data_processing_workflow_worker.workflows.predict_slate_images_workflow import (
        PredictSlateImageInput,
        SlatePredictionResult,
    )

    return PredictSlateImageInput, SlatePredictionResult


@activity.defn
async def predict_slate_image(payload):  # type: ignore[no-untyped-def]
    """Download one raw slate frame + its template PDF from Garage, run the
    board-plane estimator, and return the gated reference-point prediction."""
    payload_cls, result_cls = _models()
    if not isinstance(payload, payload_cls):
        payload = payload_cls.model_validate(payload)

    activity.logger.info(
        "predicting slate image checksum=%s image_id=%d slate_id=%d",
        payload.checksum,
        payload.image_id,
        payload.slate_id,
    )

    client = open_object_store_client()
    raw_bytes = await client.download_raw(payload.checksum)
    pdf_bytes = await client.download_slate_pdf(payload.slate_id)

    points, confidence, reason, width, height = await asyncio.to_thread(
        _predict_from_bytes,
        raw_bytes,
        pdf_bytes,
        payload.camera_matrix,
        payload.distortion_coefficients,
        payload.dpi,
        payload.template_points,
        payload.slate_name,
        DEFAULT_MIN_CONFIDENCE,
    )

    activity.logger.info(
        "predicted slate image_id=%d confidence=%.3f reason=%s dims=%dx%d",
        payload.image_id,
        confidence,
        reason,
        width,
        height,
    )
    return result_cls(
        image_id=payload.image_id,
        reference_points=points,
        confidence=confidence,
        rejected_reason=reason,
        width=width,
        height=height,
    )
