"""Module defining dive model for Fishsense API SDK."""

from datetime import datetime

from fishsense_api_sdk.models.model_base import ModelBase
from fishsense_api_sdk.models.priority import Priority


class Dive(ModelBase):
    """Model representing a dive."""

    id: int | None
    name: str | None
    path: str
    dive_datetime: datetime
    priority: Priority
    flip_dive_slate: bool | None

    # Defaulted to mirror the API, where it is
    # `Field(default=None, foreign_key="camera.id")`. Without a default here
    # the SDK cannot express *unset*, only null — and `dives.post` dumps
    # `exclude_unset`, so the difference is what reaches the wire.
    #
    # That matters because `POST /api/v1/dives/` overlays only the fields a
    # body actually sent onto the existing row. `finalize_dive_activity`
    # re-posts a dive to flip its priority and has no way to know the
    # `camera_id` preflight resolved from the MakerNote serial; it must omit
    # the field and let the overlay keep it. Sending an explicit null instead
    # is rejected with 422 `camera_id is required`.
    camera_id: int | None = None
    dive_slate_id: int | None
    # Defaulted (unlike the fields above) so consumers built before this
    # column existed — older API responses, worker test fixtures — still
    # validate. A newly-added optional column must be optional on the wire.
    calibration_dive_id: int | None = None
    # Defaulted for the same reason as `calibration_dive_id` above: a newly
    # added optional column must be optional on the wire, or every consumer
    # built against an older API response fails validation.
    notes: str | None = None
