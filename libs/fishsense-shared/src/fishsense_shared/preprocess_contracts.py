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
from typing import List, Tuple

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
    expected-laser bbox is part of the input rather than baked into
    the data-worker so the api-worker can pick a per-camera bbox if
    we ever ship more than one sensor.
    """

    dive_id: int
    image_checksums: List[str]
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]
    bbox: List[int]


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
