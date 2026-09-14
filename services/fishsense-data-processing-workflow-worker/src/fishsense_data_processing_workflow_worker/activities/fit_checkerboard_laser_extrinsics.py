"""Fit and persist `LaserExtrinsics` from checkerboard observations.

The dive-level half of checkerboard calibration, and it is deliberately the
same steps stage 13 takes once its slate observations are in hand:
`fishsense_core.laser.calibrate_laser`, then the four gates
(`check_observation_geometry`, `check_fit_self_consistency`,
`check_baseline_plausible`, `check_calibration_describes_dive`), then
`put_laser_extrinsics`. Nothing here knows the target was a board —
by the time an observation reaches this activity it is a 3-D point and the
2-D dot it came from, which is all the fit ever needed.

The threshold and the gates are imported from the stage-13 activity and
`calibration_consistency` rather than restated. `MIN_LASER_POINTS` is already
one number spelled on both sides of the worker boundary (the api's
`MIN_SLATE_LASER_POINTS` mirrors it), and a third copy that could drift from
the cohort's is the wedge shape this repo keeps rediscovering: cohort says
eligible, activity refuses, nothing is written, dive re-selected hourly
forever.
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
    CalibrationDoesNotDescribeDiveError,
    CalibrationUnderdeterminedError,
    check_baseline_plausible,
    check_calibration_describes_dive,
    check_fit_self_consistency,
    check_observation_geometry,
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


async def _record_refusal(dive_id: int, reason: str) -> None:
    """Take the dive out of the calibration cohort until its inputs change.

    Only for refusals that are deterministic in the observations this run was
    dispatched with — the same set marked non-retryable. Without it the dive is
    re-selected hourly forever, re-staging its raw `.ORF`s each time, and
    because the selector is `ORDER BY id LIMIT 1` it blocks every dive behind
    it (the prod dive-347 shape).

    Best-effort: a failure here must not mask the refusal it is annotating.
    The worst case is the pre-existing behaviour — the dive is offered again —
    which is strictly better than losing the real error.
    """
    try:
        async with get_fs_client() as fs:
            await fs.dives.set_calibration_refused(dive_id, reason)
    except Exception as exc:  # pylint: disable=broad-except
        activity.logger.error(
            "could not record calibration refusal for dive_id=%d: %s", dive_id, exc
        )


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
        # Deterministic in this run's frames — the boards were not found, and
        # re-firing cannot change that. Recorded so the dive leaves the cohort
        # instead of being re-selected hourly forever; it comes back on its own
        # if its labels are updated.
        reason = (
            f"insufficient checkerboard laser points ({len(points)} < "
            f"{MIN_LASER_POINTS}) from {len(payload.observations)} frames; "
            f"skipped={dict(sorted(skipped.items()))}"
        )
        await _record_refusal(payload.dive_id, reason)
        raise ApplicationError(
            f"dive_id={payload.dive_id}: {reason}",
            type="InsufficientCheckerboardPoints",
            non_retryable=True,
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

    # A refusal from any of the four cannot come good on a retry: three are
    # deterministic functions of the observations this run was dispatched
    # with, and the fourth re-reads the same dive labels. Left as plain
    # `ValueError`s Temporal reschedules them until the child's 2 h execution
    # timeout, holding the parent, keeping the dive's raw scratch alive and
    # skipping two hourly firings — all to re-derive the same answer. Marked
    # non-retryable they fail the child in one attempt.
    #
    # The baseline gate is the one that sees this producer's failure mode: six
    # of its calibrations fitted baselines of 2.35 to 22.22 cm against a fleet
    # constant of ~10.4 cm, and the self-consistency check passed every one,
    # because a wrong offset does not move the ray's projection.
    # The dive's own dots, for the describes-the-dive gate. Fetched before the
    # try so a transport failure stays retryable rather than being recorded as
    # a deterministic refusal.
    async with get_fs_client() as fs:
        dive_laser_labels = await fs.labels.get_laser_labels(payload.dive_id) or []
    dive_dots = np.array(
        [
            (float(label.x), float(label.y))
            for label in dive_laser_labels
            if label.x is not None and label.y is not None
        ],
        dtype=float,
    ).reshape(-1, 2)

    try:
        # Underdetermination first: both projection gates abstain on exactly
        # the degenerate geometry that most needs refusing. A board burst shot
        # at one distance determines the ray's direction no better than one
        # frame does, however many corners it detects.
        check_observation_geometry(fitted_points)
        check_fit_self_consistency(
            laser_position,
            laser_axis,
            np.array(payload.camera_matrix, dtype=float),
            np.array(dots, dtype=float),
        )
        check_baseline_plausible(laser_position)
        # And whether this fit describes the frames it will measure, which the
        # three above cannot ask: they all compare the ray against its own
        # observations.
        check_calibration_describes_dive(
            laser_position,
            laser_axis,
            np.array(payload.camera_matrix, dtype=float),
            dive_dots,
        )
    except (
        CalibrationInconsistentError,
        CalibrationImplausibleError,
        CalibrationUnderdeterminedError,
        CalibrationDoesNotDescribeDiveError,
    ) as exc:
        await _record_refusal(payload.dive_id, str(exc))
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
