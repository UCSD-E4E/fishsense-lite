"""Calibration-borrow candidate (read-only response DTO).

Wire mirror of `fishsense_api.controllers.dive_controller.CalibrationCandidate`:
a dive whose laser-line fingerprint matches the target's, so its
`LaserExtrinsics` can be borrowed. Not a persisted model — no drift-test pair.
"""

from datetime import datetime

from fishsense_api_sdk.models.model_base import ModelBase


class CalibrationCandidate(ModelBase):
    """A ranked calibration-borrow candidate for a dive."""

    dive_id: int
    name: str | None
    camera_id: int | None
    dive_datetime: datetime
    laser_extrinsics_id: int
    line_angle_deg: float
    line_offset_px: float
    line_confidence: float
    residual_std: float
    days_apart: float
