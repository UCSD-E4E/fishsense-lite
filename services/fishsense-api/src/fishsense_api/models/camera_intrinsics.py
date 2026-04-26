"""Model representing camera intrinsics for FishSense API."""

from typing import List

from sqlmodel import JSON, Column, Field, SQLModel


class CameraIntrinsics(SQLModel, table=True):
    """Model representing camera intrinsics."""

    id: int | None = Field(default=None, primary_key=True)
    camera_matrix: List[List[float]] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    distortion_coefficients: List[float] = Field(
        default_factory=list, sa_column=Column(JSON)
    )

    camera_id: int = Field(default=None, foreign_key="camera.id")
