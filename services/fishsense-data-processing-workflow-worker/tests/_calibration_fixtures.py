"""Dive-dot stand-ins shared by the two checkerboard-fit suites.

`check_calibration_describes_dive` compares a fit against **every live laser
dot in the dive**, not against the frames it was fitted from, so every test
that drives a fit activity end to end has to supply those dots. Both
checkerboard suites need the same stand-in — dots lying exactly on the
projection of the ray their fake kernel returns, i.e. "the whole dive agrees"
— so it lives here once rather than being copied into both.

Not a `conftest.py` fixture: the callers need it as a plain function, because
the ray they project is the one their own fake kernel was told to return.
"""

from __future__ import annotations

from typing import Sequence
from unittest.mock import AsyncMock, MagicMock

import numpy as np
from fishsense_api_sdk.models.laser_label import LaserLabel


def dots_on_projected_ray(
    origin: Sequence[float],
    axis: Sequence[float],
    camera_matrix: Sequence[Sequence[float]],
    depths: Sequence[float] | None = None,
) -> np.ndarray:
    """The pixels a dot on `origin` + t * `axis` would occupy, per range."""
    origin = np.asarray(origin, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    depths = np.linspace(0.6, 3.5, 40) if depths is None else np.asarray(depths)
    ts = (depths - origin[2]) / axis[2]
    points = origin[None, :] + ts[:, None] * axis[None, :]
    homogeneous = (np.asarray(camera_matrix, dtype=float) @ points.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def laser_labels_on_ray(
    origin: Sequence[float],
    axis: Sequence[float],
    camera_matrix: Sequence[Sequence[float]],
    depths: Sequence[float] | None = None,
    offset_px: float = 0.0,
) -> list[LaserLabel]:
    """SDK-shaped labels for `fs.labels.get_laser_labels`.

    `offset_px` slides the whole population perpendicular to its own long
    axis, which is how a dive that disagrees with its calibration looks: the
    dots are still collinear, just not on the fitted ray.
    """
    dots = dots_on_projected_ray(origin, axis, camera_matrix, depths)
    if offset_px:
        centred = dots - dots.mean(axis=0)
        _, _, vt = np.linalg.svd(centred)
        normal = np.array([-vt[0][1], vt[0][0]])
        dots = dots + offset_px * normal
    return [
        LaserLabel(
            id=900 + i,
            label_studio_task_id=800 + i,
            label_studio_project_id=73,
            x=float(x),
            y=float(y),
            label="laser",
            updated_at=None,
            superseded=False,
            completed=True,
            label_studio_json=None,
            image_id=700 + i,
            user_id=None,
        )
        for i, (x, y) in enumerate(dots)
    ]


def make_fit_client(
    origin: Sequence[float],
    axis: Sequence[float],
    camera_matrix: Sequence[Sequence[float]],
    *,
    put_return: int | None = 7,
    dive_offset_px: float = 0.0,
) -> MagicMock:
    """The whole SDK surface the checkerboard fit touches, as one mock.

    Three suites drive that activity end to end — the gate tripwires, the
    refusal-recording tripwires and the workflow contract tests — and each
    needs the same three calls stubbed: the write, the refusal record, and the
    dive's own laser dots. The mock doubles as its own async context manager,
    so `monkeypatch.setattr(module, "get_fs_client", lambda: client)` is all a
    caller needs.
    """
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.dives = MagicMock()
    client.dives.put_laser_extrinsics = AsyncMock(return_value=put_return)
    client.dives.set_calibration_refused = AsyncMock(return_value=None)
    client.labels = MagicMock()
    client.labels.get_laser_labels = AsyncMock(
        return_value=laser_labels_on_ray(
            origin, axis, camera_matrix, offset_px=dive_offset_px
        )
    )
    return client
