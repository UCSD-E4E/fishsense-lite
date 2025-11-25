"""Module defining dive frame cluster model for Fishsense API SDK."""

from datetime import datetime
from typing import List

from pydantic import BaseModel

from fishsense_api_sdk.models.data_source import DataSource


class DiveFrameCluster(BaseModel):
    """Model representing a cluster of frames within a dive."""

    id: int
    image_ids: List[int]
    data_source: DataSource
    updated_at: datetime | None

    dive_id: int | None
