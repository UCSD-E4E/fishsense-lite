"""Models for Dive Frame Clusters in FishSense API."""

from datetime import datetime
from typing import List

from pydantic import BaseModel
from sqlmodel import Column, DateTime, Enum, Field

from fishsense_api.models.data_source import DataSource
from fishsense_api.models.model_base import ModelBase


class DiveFrameCluster(ModelBase, table=True):
    """Model representing a cluster of frames within a dive."""

    id: int | None = Field(default=None, primary_key=True)
    data_source: DataSource = Field(sa_column=Column(Enum(DataSource)))
    updated_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)

    dive_id: int | None = Field(default=None, foreign_key="dive.id")
    fish_id: int | None = Field(default=None, foreign_key="fish.id")


class DiveFrameClusterImageMapping(ModelBase, table=True):
    """Association table mapping images to dive frame clusters."""

    dive_frame_cluster_id: int = Field(
        default=None, foreign_key="diveframecluster.id", primary_key=True
    )
    image_id: int = Field(default=None, foreign_key="image.id", primary_key=True)


class DiveFrameClusterJson(BaseModel):
    """Pydantic model for serializing DiveFrameCluster data."""

    id: int | None
    image_ids: List[int]
    data_source: DataSource
    updated_at: datetime | None

    dive_id: int | None
    fish_id: int | None
