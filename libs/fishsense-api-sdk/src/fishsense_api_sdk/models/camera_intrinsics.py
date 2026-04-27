"""Module defining camera intrinsics model for Fishsense API SDK."""

from typing import List, Self

import numpy as np
from pydantic import BaseModel


class _CameraIntrinsics(BaseModel):
    """Model representing camera intrinsics."""

    id: int | None
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]

    camera_id: int


class CameraIntrinsics:
    # pylint: disable=too-few-public-methods
    """Class representing camera intrinsics."""

    def __init__(
        self,
        camera_matrix: "np.ndarray[float] | None",
        distortion_coefficients: "np.ndarray[float] | None",
        camera_id: int | None,
        id: int | None = None,  # pylint: disable=redefined-builtin
    ):
        self.id = id
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = (
            distortion_coefficients.squeeze()
            if distortion_coefficients is not None
            else None
        )
        self.camera_id = camera_id

    @staticmethod
    def _from_internal(internal: _CameraIntrinsics) -> Self:
        return CameraIntrinsics(
            id=internal.id,
            camera_matrix=np.array(internal.camera_matrix, dtype=float),
            distortion_coefficients=np.array(
                internal.distortion_coefficients, dtype=float
            ),
            camera_id=internal.camera_id,
        )

    def _to_internal(self) -> _CameraIntrinsics:
        return _CameraIntrinsics(
            id=self.id,
            camera_matrix=(
                self.camera_matrix.tolist() if self.camera_matrix is not None else []
            ),
            distortion_coefficients=(
                self.distortion_coefficients.tolist()
                if self.distortion_coefficients is not None
                else []
            ),
            camera_id=self.camera_id,
        )
