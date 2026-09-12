"""Fit and persist `LaserExtrinsics` from checkerboard observations.

The dive-level half of checkerboard calibration, and it is deliberately the
same three steps stage 13 takes once its slate observations are in hand:
`fishsense_core.laser.calibrate_laser`, then `check_fit_self_consistency`,
then `put_laser_extrinsics`. Nothing here knows the target was a board —
by the time an observation reaches this activity it is a 3-D point and the
2-D dot it came from, which is all the fit ever needed.

The threshold and the gate are imported from the stage-13 activity rather than
restated. `MIN_LASER_POINTS` is already one number spelled on both sides of
the worker boundary (the api's `MIN_SLATE_LASER_POINTS` mirrors it), and a
third copy that could drift from the cohort's is the wedge shape this repo
keeps rediscovering: cohort says eligible, activity refuses, nothing is
written, dive re-selected hourly forever.
"""

from __future__ import annotations

from collections import Counter
from typing import List

import numpy as np
from fishsense_api_sdk.models.laser_extrinsics import LaserExtrinsics
from fishsense_core.laser import calibrate_laser as _calibrate_laser
from fishsense_shared import CheckerboardObservation
from temporalio import activity
from temporalio.exceptions import ApplicationError

from fishsense_data_processing_workflow_worker.activities.perform_laser_calibration_activity import (  # noqa: E501  pylint: disable=line-too-long
    MIN_LASER_POINTS,
)
from fishsense_data_processing_workflow_worker.activities.utils import get_fs_client
from fishsense_data_processing_workflow_worker.calibration_consistency import (
    CalibrationImplausibleError,
    CalibrationInconsistentError,
    check_baseline_plausible,
    check_fit_self_consistency,
)
from fishsense_data_processing_workflow_worker.robust_laser_fit import (
    trim_outlying_observations,
)

__all__ = ["fit_checkerboard_laser_extrinsics"]


def _input_model():
    from fishsense_data_processing_workflow_worker.workflows import (
        perform_checkerboard_calibration_workflow as workflow,
    )

    return workflow.FitCheckerboardExtrinsicsInput


def _usable(
    observations: List[CheckerboardObservation],
) -> tuple[list[list[float]], list[tuple[float, float]]]:
    """Split the observations that produced a point into points and dots.

    Returned in lockstep: the 2-D dots feed the self-consistency gate, which
    asks whether the fitted ray reprojects onto the very dots it came from.
    """
    points: list[list[float]] = []
    dots: list[tuple[float, float]] = []
    for observation in observations:
        if observation.point is None:
            continue
        points.append([float(value) for value in observation.point])
        dots.append((float(observation.laser_x), float(observation.laser_y)))
    return points, dots


@activity.defn
async def fit_checkerboard_laser_extrinsics(payload) -> int:
    """Fit `LaserExtrinsics` for the dive and persist it. Returns the row id.

    Raises when fewer than `MIN_LASER_POINTS` frames yielded a usable
    observation. That is a real data problem worth surfacing — the cohort
    promised at least that many laser-dotted frames, so falling short means
    the boards themselves were not found, and no amount of re-firing will
    change that. The remedy is operator-side (clear the dive's calibration
    target, or park the dive), which is why the cohort's docstring names it.

    Never persists a fit that disagrees with its own dots:
    `check_fit_self_consistency` raises instead. That gate is not belt and
    braces — a mixed dot population shipped a calibration whose length errors
    reached +137% downstream on prod dive 77.
    """
    payload_cls = _input_model()
    if not isinstance(payload, payload_cls):
        payload = payload_cls.model_validate(payload)

    points, dots = _usable(payload.observations)
    skipped = Counter(
        o.skip_reason or "unknown" for o in payload.observations if o.point is None
    )
    activity.logger.info(
        "checkerboard calibration dive_id=%d usable=%d of %d observations "
        "skipped=%s",
        payload.dive_id,
        len(points),
        len(payload.observations),
        dict(sorted(skipped.items())) or "{}",
    )
    if len(points) < MIN_LASER_POINTS:
        raise ValueError(
            f"dive_id={payload.dive_id}: insufficient checkerboard laser points "
            f"({len(points)} < {MIN_LASER_POINTS}) from "
            f"{len(payload.observations)} frames; skipped="
            f"{dict(sorted(skipped.items()))}"
        )

    # Same trim as stage 13, for the same reason: `calibrate_laser` has no
    # outlier rejection, and the z=0 crossing it reports levers a small angular
    # error into a large baseline error.
    fitted_points = trim_outlying_observations(np.array(points))
    if len(fitted_points) < len(points):
        activity.logger.info(
            "dive_id=%d: trimmed %d of %d checkerboard observations as outliers",
            payload.dive_id,
            len(points) - len(fitted_points),
            len(points),
        )
    origin, orientation = _calibrate_laser(fitted_points.astype(np.float32))
    # Rust kernel returns origin with z=0 implicit; pad to a 3-vector to match
    # the LaserExtrinsics SDK surface. Same as stage 13.
    laser_position = np.array([float(origin[0]), float(origin[1]), 0.0], dtype=float)
    laser_axis = np.asarray(orientation, dtype=float)

    # Both gates are deterministic functions of the observations this run was
    # dispatched with, so a refusal cannot come good on a retry. Left as plain
    # `ValueError`s Temporal reschedules them until the child's 2 h execution
    # timeout, holding the parent, keeping the dive's raw scratch alive and
    # skipping two hourly firings — all to re-derive the same answer. Marked
    # non-retryable they fail the child in one attempt.
    #
    # The baseline gate is the one that sees this producer's failure mode: six
    # of its calibrations fitted baselines of 2.35 to 22.22 cm against a fleet
    # constant of ~10.4 cm, and the self-consistency check passed every one,
    # because a wrong offset does not move the ray's projection.
    try:
        check_fit_self_consistency(
            laser_position,
            laser_axis,
            np.array(payload.camera_matrix, dtype=float),
            np.array(dots, dtype=float),
        )
        check_baseline_plausible(laser_position)
    except (CalibrationInconsistentError, CalibrationImplausibleError) as exc:
        raise ApplicationError(
            f"dive_id={payload.dive_id}: {exc}",
            type=type(exc).__name__,
            non_retryable=True,
        ) from exc

    async with get_fs_client() as fs:
        return await fs.dives.put_laser_extrinsics(
            payload.dive_id,
            LaserExtrinsics(
                laser_position=laser_position,
                laser_axis=laser_axis,
                dive_id=payload.dive_id,
                camera_id=payload.camera_id,
            ),
        )
