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

from typing import List

from pydantic import BaseModel


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
