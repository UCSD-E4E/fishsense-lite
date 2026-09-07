"""SDK mirror of the API's `CalibrationTarget` table.

A planar target of known geometry that laser extrinsics can be fitted
against — the checkerboard counterpart of `DiveSlate`. `rows` and `cols` are
**interior corners**, not squares, and `square_size_m` is the measured grid
pitch in metres. See the API model for why that number is required rather
than nullable.
"""

from datetime import datetime

from fishsense_api_sdk.models.model_base import ModelBase


class CalibrationTarget(ModelBase):
    """Model representing a planar calibration target."""

    id: int | None
    name: str
    rows: int
    cols: int
    square_size_m: float
    notes: str | None
    created_at: datetime | None
