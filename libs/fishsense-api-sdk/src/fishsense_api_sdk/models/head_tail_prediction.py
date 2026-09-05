"""Module defining head/tail prediction model for Fishsense API SDK.

Mirrors fishsense-api's `HeadTailPrediction` SQLModel — model-predicted
snout/fork keypoints for an image, in rectified-image pixels (already lifted
out of the laser-centred crop the model ran on). All four coordinates are None
on an abstention; `status` says which kind.
"""

from datetime import datetime

from fishsense_api_sdk.models.model_base import ModelBase


class HeadTailPrediction(ModelBase):
    """Model representing predicted snout/fork keypoints."""

    id: int | None = None
    head_x: float | None = None
    head_y: float | None = None
    tail_x: float | None = None
    tail_y: float | None = None
    # Rectified frame dimensions the coordinates are relative to (for the pixel
    # -> Label Studio keypoint-percentage conversion in head/tail populate).
    width: int | None = None
    height: int | None = None
    # Which fish, and how fish-shaped it was; the silhouette band is the
    # confidence gate, applied at seed time so it stays retunable. See the API
    # SQLModel.
    mask_area_px: int | None = None
    silhouette_ratio: float | None = None
    # Origin of the laser-centred crop the mask was found in — provenance for
    # coordinates that are already lifted back to the full frame.
    crop_x: int | None = None
    crop_y: int | None = None
    # The laser label that selected the fish, so the cohort can select on
    # mismatch when RANSAC later supersedes it.
    laser_label_id: int | None = None
    # Stage version that produced this row (the cohort selects on a mismatch),
    # plus the provenance recorded beside it. See the API SQLModel.
    predictor_version: int | None = None
    checkpoint: str | None = None
    core_version: str | None = None
    status: str = "predicted"
    rejected_low_confidence: bool = False
    created_at: datetime | None = None
    image_id: int | None = None
