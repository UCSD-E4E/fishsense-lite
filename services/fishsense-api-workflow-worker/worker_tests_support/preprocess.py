"""Fixtures shared by the resolver-activity suites.

`resolve_laser/species/headtail/slate_preprocess_inputs_activity` answer the
same question for four stages, and their tests were four copies of the same
SDK-model builders and the same `MagicMock` client. `duplicate-code` flagged
six pairs of them.

Kept separate from `populate.py`: those suites build images on a different
dive and camera, and forcing one builder to serve both would mean a pile of
keyword arguments whose only job is to say which suite is calling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel

# Olympus-scale intrinsics on a 4096x3072 frame — real enough that any
# projection a resolver performs lands in a plausible range.
CAMERA_MATRIX = np.array(
    [[3000.0, 0.0, 2048.0], [0.0, 3000.0, 1536.0], [0.0, 0.0, 1.0]]
)
DISTORTION = np.array([-0.05, 0.01, 0.0, 0.0, 0.0])

__all__ = ["CAMERA_MATRIX", "DISTORTION", "dive", "image", "laser_label", "intrinsics"]


def dive(dive_id: int = 42, *, camera_id: Optional[int] = 1) -> Dive:
    return Dive(
        id=dive_id,
        name=f"dive-{dive_id}",
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority="HIGH",
        flip_dive_slate=False,
        camera_id=camera_id,
        dive_slate_id=None,
    )


def image(image_id: int, checksum: str) -> Image:
    return Image(
        id=image_id,
        path=f"/dev/null/{image_id}",
        taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        checksum=checksum,
        is_canonical=True,
        dive_id=42,
        camera_id=1,
    )


def laser_label(
    image_id: int,
    *,
    completed: bool,
    project_id: Optional[int] = 73,
    needs_reprocess: bool = False,
) -> LaserLabel:
    return LaserLabel(
        id=None,
        image_id=image_id,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=project_id,
        updated_at=None,
        completed=completed,
        label_studio_json={},
        user_id=None,
        superseded=False,
        needs_reprocess=needs_reprocess,
        x=None,
        y=None,
        label=None,
    )


def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_matrix=CAMERA_MATRIX,
        distortion_coefficients=DISTORTION,
        camera_id=1,
    )
