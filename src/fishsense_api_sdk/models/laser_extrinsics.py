"""LaserExtrinsics model for FishSense API SDK."""

from datetime import datetime
from typing import List, Self

import numpy as np

from fishsense_api_sdk.models.model_base import ModelBase


class _LaserExtrinsics(ModelBase):
    id: int | None
    laser_position: List[float]
    laser_axis: List[float]
    created_at: datetime | None

    dive_id: int | None
    camera_id: int


class LaserExtrinsics:
    """LaserExtrinsics model representing laser extrinsics information."""

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        laser_position: np.ndarray[float] | None,
        laser_axis: np.ndarray[float] | None,
        dive_id: int | None,
        camera_id: int | None,
        id: int | None = None,  # pylint: disable=redefined-builtin
        created_at: datetime | None = None,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.id = id
        self.laser_position = (
            laser_position.squeeze() if laser_position is not None else None
        )
        self.laser_axis = laser_axis.squeeze() if laser_axis is not None else None
        self.created_at = created_at
        self.dive_id = dive_id
        self.camera_id = camera_id

    @staticmethod
    def _from_internal(internal: _LaserExtrinsics) -> Self:
        return LaserExtrinsics(
            id=internal.id,
            laser_position=np.array(internal.laser_position, dtype=float),
            laser_axis=np.array(internal.laser_axis, dtype=float),
            created_at=internal.created_at,
            dive_id=internal.dive_id,
            camera_id=internal.camera_id,
        )

    def _to_internal(self) -> _LaserExtrinsics:
        return _LaserExtrinsics(
            id=self.id,
            laser_position=(
                self.laser_position.tolist()
                if self.laser_position is not None
                else None
            ),
            laser_axis=(
                self.laser_axis.tolist() if self.laser_axis is not None else None
            ),
            created_at=self.created_at,
            dive_id=self.dive_id,
            camera_id=self.camera_id,
        )
