"""Module defining species label model for Fishsense API SDK."""

from datetime import datetime
from typing import Any, Dict

from fishsense_api_sdk.models.model_base import ModelBase


class SpeciesLabel(ModelBase):
    """Model representing a species label."""

    id: int | None
    label_studio_task_id: int | None
    label_studio_project_id: int | None
    image_url: str | None
    updated_at: datetime | None
    completed: bool | None
    superseded: bool | None
    needs_reprocess: bool = False
    grouping: str | None
    top_three_photos_of_group: bool | None
    slate_upside_down: bool | None
    laser_x: float | None
    laser_y: float | None
    laser_label: str | None
    content_of_image: str | None
    fish_measurable_category: str | None
    fish_angle_category: str | None
    # Mirrors the API column: the reviewed angle in degrees, distinct from the
    # Label Studio category whose `x > 15°` bucket cannot separate one
    # commanded angle from another. The SDK must carry it so a read-modify-write
    # caller round-trips it rather than dropping it from the body.
    # Defaulted, unlike its neighbours, so that constructing a SpeciesLabel
    # without an angle stays legal — every existing writer builds this model
    # field-by-field and none of them knows about the angle. It also means the
    # field is absent from `model_fields_set` for those writers, which is what
    # `_upsert_label` keys "unmentioned" off when it preserves stored values.
    fish_angle_degrees: float | None = None
    fish_curved_category: str | None
    label_studio_json: Dict[str, Any] | str | None

    image_id: int | None
    user_id: int | None
