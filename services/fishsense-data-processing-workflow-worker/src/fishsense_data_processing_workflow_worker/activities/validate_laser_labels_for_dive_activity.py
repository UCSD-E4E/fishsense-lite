"""Per-dive laser-label validation (observe-only).

Fetches the dive's non-superseded `LaserLabel`s from fishsense-api,
fits a RANSAC line through the positives, and logs any labels whose
perpendicular distance exceeds the per-dive outlier threshold. The
laser rig is fixed across a dive, so all positive laser observations
should be colinear in image space — outliers are likely mislabeled.

Phase 1 (this commit): structured logging only. No `superseded`
writes. Once the threshold is calibrated against real prod label
distributions, a follow-up commit flips the writeback on. See the
"Open follow-ups" entry in the repo-root CLAUDE.md.
"""

from __future__ import annotations

from typing import List

import numpy as np
from fishsense_api_sdk.models.laser_label import LaserLabel
from temporalio import activity

from fishsense_data_processing_workflow_worker.activities.utils import get_fs_client
from fishsense_data_processing_workflow_worker.laser_label_validation.line_fit import (
    MIN_POINTS_FOR_LINE,
    fit_dive_line,
    flag_outliers,
)

__all__ = ["validate_laser_labels_for_dive_activity"]


def _positive_xy(labels: List[LaserLabel]) -> tuple[np.ndarray, List[LaserLabel]]:
    """Pull the positive (x, y) labels and the matching label objects.

    A "positive" is a laser-localization label with both coordinates set.
    Sentinel rows seeded by populate (no laser visible) and skipped
    annotations land here as null x/y and are excluded.
    """
    positives = [
        label for label in labels if label.x is not None and label.y is not None
    ]
    if not positives:
        return np.empty((0, 2), dtype=float), []
    xy = np.array(
        [(float(label.x), float(label.y)) for label in positives], dtype=float
    )
    return xy, positives


@activity.defn
async def validate_laser_labels_for_dive_activity(dive_id: int) -> int:
    """Run RANSAC line-fit validation for `dive_id`. Returns the number
    of labels flagged as outliers (0 when the line isn't confident or
    there aren't enough positives to fit).

    Observe-only: results are logged via `activity.logger`; no
    `superseded` writes happen.

    Heartbeats around the SDK fetch + each compute milestone so a
    stalled fetch (large dive's `label_studio_json` payload over
    Traefik) trips `heartbeat_timeout` instead of grinding all the way
    to `schedule_to_close_timeout` with no signal of where it hung.
    """
    activity.logger.info(
        "dive_id=%d validation starting; fetching laser labels", dive_id
    )
    activity.heartbeat()
    async with get_fs_client() as fs:
        labels = await fs.labels.get_laser_labels(dive_id) or []
    activity.logger.info(
        "dive_id=%d fetched %d laser label rows", dive_id, len(labels)
    )
    activity.heartbeat()

    if not labels:
        activity.logger.info(
            "dive_id=%d has no laser labels; skipping validation", dive_id
        )
        return 0

    xy, positives = _positive_xy(labels)
    if xy.shape[0] < MIN_POINTS_FOR_LINE:
        activity.logger.info(
            "dive_id=%d has %d positive laser labels (<%d); "
            "skipping line fit",
            dive_id,
            xy.shape[0],
            MIN_POINTS_FOR_LINE,
        )
        return 0

    fit = fit_dive_line(xy)
    if fit is None:
        activity.logger.info(
            "dive_id=%d: line fit returned None despite %d positives "
            "(unexpected; check inputs)",
            dive_id,
            xy.shape[0],
        )
        return 0

    activity.logger.info(
        "dive_id=%d line fit: n=%d inliers=%d (%.0f%%) "
        "residual_std=%.2fpx label_noise_mad=%.2fpx "
        "line_confidence=%.1f confident=%s",
        dive_id,
        fit.n_points,
        fit.inlier_count,
        100.0 * fit.inlier_fraction,
        fit.residual_std,
        fit.label_noise_mad,
        fit.line_confidence,
        fit.is_confident,
    )

    outlier_mask = flag_outliers(xy, fit)
    n_outliers = int(outlier_mask.sum())
    if n_outliers == 0:
        activity.logger.info("dive_id=%d: no outlier laser labels", dive_id)
        return 0

    perp = fit.perpendicular_distance(xy[:, 0], xy[:, 1])
    for i, is_outlier in enumerate(outlier_mask):
        if not is_outlier:
            continue
        label = positives[i]
        activity.logger.info(
            "dive_id=%d OUTLIER laser_label_id=%s image_id=%s "
            "x=%.1f y=%.1f perp=%.2fpx label_studio_task_id=%s "
            "label_studio_project_id=%s "
            "(would set superseded=True if writeback were enabled)",
            dive_id,
            label.id,
            label.image_id,
            float(label.x),
            float(label.y),
            float(perp[i]),
            label.label_studio_task_id,
            label.label_studio_project_id,
        )

    activity.logger.info(
        "dive_id=%d flagged %d/%d positive laser labels as outliers",
        dive_id,
        n_outliers,
        xy.shape[0],
    )
    return n_outliers
