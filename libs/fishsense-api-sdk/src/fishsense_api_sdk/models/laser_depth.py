"""Module defining the per-image laser depth model for the Fishsense API SDK.

Mirrors fishsense-api's `LaserDepth` SQLModel — how far away an image's laser
dot was. `depth_m` is the Z component (the depth stage 14 back-projects head
and tail against); `range_m` is the Euclidean distance to the dot. They are
not the same number, which is why both are carried rather than one being left
for the caller to infer.

`laser_label_id` and `laser_extrinsics_id` name the inputs the value was
derived from, so a consumer can tell whether it still reflects the current
label and calibration, and `residual_m` says how well those inputs actually
agreed.
"""

from datetime import datetime

from fishsense_api_sdk.models.model_base import ModelBase


class LaserDepth(ModelBase):
    """Model representing the distance to an image's laser dot, in metres."""

    id: int | None = None
    depth_m: float
    range_m: float | None = None
    # Closest-approach distance between the camera ray and the laser ray, in
    # metres — ~0 when the dot is consistent with the calibration. A quality
    # signal, not a gate; see the API model for why it is necessary but not
    # sufficient.
    residual_m: float | None = None
    created_at: datetime | None = None
    image_id: int | None = None
    laser_label_id: int | None = None
    laser_extrinsics_id: int | None = None
