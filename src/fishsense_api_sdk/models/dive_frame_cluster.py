"""Module defining dive frame cluster model for Fishsense API SDK."""

from datetime import datetime
from typing import List

from fishsense_api_sdk.models.data_source import DataSource
from fishsense_api_sdk.models.model_base import ModelBase


class DiveFrameCluster(ModelBase):
    """Model representing a cluster of frames within a dive."""

    id: int | None
    image_ids: List[int]
    data_source: DataSource
    updated_at: datetime | None

    dive_id: int | None
    fish_id: int | None
