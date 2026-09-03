"""Workflow-input DTOs that cross worker boundaries.

api-worker parents (selection + resolution) construct these and hand
them to data-worker child workflows that do the heavy CPU work. The
shapes match each thin data-worker workflow's `run(payload)` signature
1:1 — adding a field here means the data-worker workflow can use it,
adding one only on the data-worker workflow means the api-worker
parent can't populate it.

Per-image input DTOs stay in the data-worker workflow modules — those
are internal to the fan-out and not meant for cross-worker
construction.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from pydantic import BaseModel

ReferencePoint = Tuple[float, float]


class ClusterDiveFrameImage(BaseModel):
    """Per-image timestamp pair for stage-1 clustering."""

    image_id: int
    taken_datetime: datetime


class ClusterDiveFramesInput(BaseModel):
    """Stage 1 (dive-frame clustering) workflow-level input.

    The kernel only needs `(image_id, taken_datetime)` — image bytes
    are never read, so this DTO replaces the previous
    `Iterable[Image]` shape that pulled the data-worker's local
    pydantic Image model into the cross-worker contract.
    """

    dive_id: int
    images: List[ClusterDiveFrameImage]


class PreprocessLaserImagesInput(BaseModel):
    """Stage 0.1 (laser preprocess) workflow-level input.

    Constructed by the api-worker parent (selector + resolver), passed
    to the data-worker `PreprocessLaserImagesWorkflow` child. The
    expected-laser region is part of the input rather than baked into
    the data-worker so the api-worker can pick a per-camera one if we
    ever ship more than one sensor.

    `laser_region` is the real shape — a convex polygon of `[x, y]`
    vertices in rectified pixels, drawn in the order given. `bbox` is
    its bounding box, carried alongside rather than replaced because
    the two workers deploy independently and often days apart (in-slot
    converge vs. `kubectl apply` on NRP), so both directions of the
    skew have to render something correct:

    * new api-worker -> old data-worker: pydantic ignores the unknown
      `laser_region` and the old code draws `bbox`, a superset.
    * old api-worker -> new data-worker: `laser_region` is absent, so
      the new code falls back to `bbox`, which is all the old resolver
      ever sent.

    Once every data-worker in the fleet is past the polygon, `bbox`
    can go -- that is a deliberate follow-up, not a cleanup to fold
    into an unrelated change.
    """

    dive_id: int
    image_checksums: List[str]
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]
    bbox: List[int]
    laser_region: Optional[List[List[int]]] = None


class PreprocessSpeciesImagesInput(BaseModel):
    """Stage 2 (species preprocess) workflow-level input.

    Clusters preserve the temporal grouping from
    `DiveFrameCluster(data_source=PREDICTION)` so the per-image overlay
    can render "image i of N" for each cluster. Cluster image_ids are
    pre-filtered by the api-worker resolver to images with a valid
    laser label and no non-sentinel species label.
    """

    dive_id: int
    clusters: List[List[str]]  # each inner list is a PREDICTION cluster of checksums
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]


class PreprocessHeadtailImagesInput(BaseModel):
    """Stage 5.1 (head/tail preprocess) workflow-level input.

    Image set is filtered to species labels with
    `top_three_photos_of_group=True` whose head/tail label is not yet
    complete — same predicate `populate_headtail_label_studio_project_activity`
    uses, so populate consumes exactly what preprocess produces.
    """

    dive_id: int
    image_checksums: List[str]
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]


class PredictLaserImage(BaseModel):
    """Per-image (checksum, image_id) pair for laser prediction.

    Both are needed: the checksum fetches the raw bytes from Garage, and
    the image_id is what the prediction result is keyed back to so the
    api-worker can persist it against the right image.
    """

    image_id: int
    checksum: str


class PredictLaserImagesInput(BaseModel):
    """Laser-detector (model-assisted labeling) workflow-level input.

    Constructed by the api-worker parent (selector + resolver), passed to
    the GPU data-worker `PredictLaserImagesWorkflow` child. The fishsense-core
    `LaserDetector` rectifies its output into camera-corrected pixels using
    the dive's intrinsics, so predictions land in the same space labelers
    place `LaserLabel.x/y`.
    """

    dive_id: int
    images: List[PredictLaserImage]
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]
    # Laser wavelength ("red" / "green"); None when the dive's laser color
    # isn't known — the model uses an "unknown" wavelength channel then.
    wavelength: str | None = None
    # Convex polygon of [x, y] rectified pixels outside which a predicted dot
    # is not believed. Optional so an api-worker that predates the gate still
    # produces a valid payload for a newer data-worker, which then does not
    # gate — the same rolling-deploy reasoning as
    # `PreprocessLaserImagesInput.laser_region`. See
    # `fishsense_shared.laser_region`.
    laser_region: Optional[List[List[int]]] = None


class LaserPredictionResult(BaseModel):
    """One image's predicted laser dot, returned by the data-worker
    `PredictLaserImagesWorkflow` and persisted by the api-worker parent.

    In rectified-image pixels (the space labelers place `LaserLabel.x/y`).
    `x`/`y` are None when the detector found no laser; `confidence` is always
    reported. `width`/`height` are the rectified frame dimensions the x/y are
    relative to — the laser populate step needs them to convert pixels to the
    percentages Label Studio keypoints use. Cross-worker, so it lives here
    rather than in the data-worker workflow module.
    """

    image_id: int
    x: float | None
    y: float | None
    confidence: float
    width: int | None = None
    height: int | None = None
    # "red" / "green" sampled from the dot's own pixels, or None when there is
    # no dot to sample. Advisory per image: laser color is a per-dive constant
    # in practice (143 prod dives all-red, 88 all-green, and the 31 "mixed"
    # ones carry a 1.2% minority that is labeler slips), so populate takes the
    # dive-level majority rather than trusting any single frame.
    color: str | None = None
    # Signed strength of that call, in 8-bit levels: positive is redder, and
    # the magnitude is how far apart the channels were. Carried so a close
    # call can be recognised as one rather than silently counting as a full
    # vote.
    color_margin: float | None = None
    # True when the detector *did* find a dot but it fell outside
    # `laser_region`, so x/y were dropped. Distinct from an ordinary
    # non-detection, which is the model finding nothing at all — without this
    # the two are indistinguishable downstream and a mis-sized region would
    # look like a model that stopped working.
    rejected_out_of_region: bool = False
    # Stage version that produced this result, and the provenance recorded
    # beside it. Stamped by the data-worker (which runs the detector) from the
    # shared constant both workers import, so a rolling deploy where the two
    # disagree costs at most one extra round of re-prediction.
    predictor_version: int | None = None
    checkpoint: str | None = None
    core_version: str | None = None


class PredictSlateImage(BaseModel):
    """Per-image (checksum, image_id) pair for slate prediction.

    The checksum fetches the raw bytes from Garage; the image_id keys the
    prediction result back so the api-worker persists it against the right image.
    """

    image_id: int
    checksum: str


class PredictSlateImagesInput(BaseModel):
    """Slate-detector (model-assisted labeling) workflow-level input.

    Constructed by the api-worker parent (selector + resolver), passed to the
    CPU data-worker `PredictSlateImagesWorkflow` child. Carries the dive's slate
    template (id/name/dpi/reference points) + camera intrinsics so the
    data-worker renders the template and estimates the board without extra
    fishsense-api calls.
    """

    dive_id: int
    slate_id: int
    slate_name: str
    dpi: float
    template_points: List[List[float]]
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]
    images: List[PredictSlateImage]


class SlatePredictionResult(BaseModel):
    """One frame's gated slate prediction, returned by the data-worker
    `PredictSlateImagesWorkflow` and persisted by the api-worker parent.

    `reference_points` are in rectified-photo pixels (the space the sync
    activity stores `DiveSlateLabel.reference_points` after stripping the
    composite panel offset), or None when the estimate was rejected — see
    `rejected_reason`. Cross-worker, so it lives here rather than in the
    data-worker workflow module.
    """

    image_id: int
    reference_points: List[List[float]] | None = None
    confidence: float = 0.0
    rejected_reason: str | None = None
    width: int = 0
    height: int = 0


class PreprocessSlateImagesInput(BaseModel):
    """Stage 9 (slate preprocess) workflow-level input.

    Slate metadata travels alongside the image set so the data-worker
    can render the PDF-composite overlay without an extra
    fishsense-api call.
    """

    dive_id: int
    image_checksums: List[str]
    slate_id: int
    slate_dpi: int
    reference_points: List[ReferencePoint]
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]


class PredictHeadtailImage(BaseModel):
    """Per-image input for head/tail prediction.

    Carries the laser dots as well as the checksum, because the gate is also
    the crop centre: the predictor looks only at a window centred on the dot
    rather than searching the frame (see docs/plans/headtail-prediction.md
    §0.2b). `laser_label_ids` is parallel to `laser_points`, so the result can
    name which dot chose the fish and the cohort can later select on that dot
    having been superseded.

    An image may carry more than one valid laser label — 461 prod images do —
    and first-hit-wins is what was measured.
    """

    image_id: int
    checksum: str
    laser_points: List[List[float]]
    laser_label_ids: List[int]
    # Which JPEG prefix to read. Set by the workflow from its own
    # `jpeg_folder`, so the physical key contract stays owned by
    # `fishsense_shared.object_store` and the activity never hard-codes a
    # prefix or reaches into worker config for one.
    jpeg_folder: str = ""


class PredictHeadtailImagesInput(BaseModel):
    """Head/tail predict workflow-level input (api-worker -> data-worker).

    No camera intrinsics and no raw bytes: this stage reads the stage-5.1 JPEG
    that already exists in Garage, which is the exact frame the labeler sees.
    `jpeg_folder` is a parameter rather than a constant so the physical key
    contract stays owned by `fishsense_shared.object_store`.
    """

    dive_id: int
    images: List[PredictHeadtailImage]
    jpeg_folder: str


class HeadtailPredictionResult(BaseModel):
    """Per-image head/tail prediction (data-worker -> api-worker).

    Coordinates are rectified-frame pixels, already lifted out of the crop by
    `crop_x`/`crop_y` — the same space as `LaserLabel.x/y` and the labeler's
    own clicks. All four are None on an abstention, and `status` says which
    kind: "no_detections", "laser_off_all_fish" or "headtail_failed".
    """

    image_id: int
    status: str
    head_x: Optional[float] = None
    head_y: Optional[float] = None
    tail_x: Optional[float] = None
    tail_y: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    mask_area_px: Optional[int] = None
    silhouette_ratio: Optional[float] = None
    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    laser_label_id: Optional[int] = None
    predictor_version: Optional[int] = None
    checkpoint: Optional[str] = None
    core_version: Optional[str] = None
