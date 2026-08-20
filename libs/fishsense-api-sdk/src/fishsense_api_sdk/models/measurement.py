"""Measurement model for FishSense API SDK."""

from pydantic import BaseModel


class Measurement(BaseModel):
    """Measurement model representing fish measurement information."""

    id: int | None
    length_m: float | None

    image_id: int | None
    fish_id: int | None
    # Which calibration produced this length — the API stamps it so stage 14
    # can tell a current measurement from one computed against extrinsics that
    # have since been replaced. NULL on rows written before it existed.
    laser_extrinsics_id: int | None = None
