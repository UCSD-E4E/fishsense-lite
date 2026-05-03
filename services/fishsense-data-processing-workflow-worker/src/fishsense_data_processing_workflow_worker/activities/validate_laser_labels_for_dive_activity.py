"""Per-dive laser-label validation.

Fetches the dive's non-superseded `LaserLabel`s from fishsense-api,
fits a RANSAC line through the positives, and supersedes any label
whose perpendicular distance exceeds the per-dive outlier threshold.
The laser rig is fixed across a dive, so all positive laser
observations should be colinear in image space — outliers are
mislabeled and shouldn't feed downstream stage 13 calibration / stage
14 measurement.

Phase 2 (writeback enabled): each flagged label is updated with
`superseded=True` via `put_laser_label`. The endpoint is an upsert by
primary key, so re-runs on a dive whose outliers have already been
superseded are no-ops at the SDK level — `get_laser_labels` filters
on `superseded=False` server-side, so the second run sees a smaller
population, refits the line, and may flag additional borderline
labels that are now visible as outliers relative to the cleaned
inlier set. That iterative tightening is intentional.

Note on labeler corrections: once a `LaserLabel` row is superseded,
`get_laser_label_by_label_studio_id` filters it out, so a labeler
re-opening the same Label Studio task and saving a corrected position
will NOT propagate back to the DB through the existing sync path.
This is the same dead-letter semantic that `superseded` has always
had; reviving a superseded label requires an explicit operator action
(or a future workflow that diffs LS state against superseded rows).
"""

from __future__ import annotations

import asyncio
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

__all__ = ["validate_laser_labels_for_dive_activity", "SUPERSEDE_CONCURRENCY"]

# Bound on concurrent supersede PUTs per dive. A dive with many flagged
# outliers was blowing `start_to_close` (10m) on sequential PUTs because
# the SDK's 10s-timeout × 3-retry ladder per PUT compounds linearly. With
# a cap of 8, the budget is N/8 × per-PUT cost — comfortable for any
# realistic outlier count. The cap also keeps a single dive from hogging
# every outbound HTTP slot when multiple validate workflows run in
# parallel against different dives.
SUPERSEDE_CONCURRENCY = 8


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
    """Run RANSAC line-fit validation for `dive_id` and supersede any
    flagged outliers. Returns the number of labels superseded (0 when
    the line isn't confident or there aren't enough positives to fit).

    Heartbeats around the SDK fetch + each compute milestone + each
    supersede write so a stalled call (large dive's `label_studio_json`
    payload over Traefik, slow PUT, etc.) trips `heartbeat_timeout`
    instead of grinding to `schedule_to_close_timeout` with no signal
    of where it hung.

    Failure semantics: if any individual `put_laser_label` raises, the
    activity raises and Temporal retries the whole activity. The
    activity is idempotent at the dive level — already-superseded
    labels are filtered out by `get_laser_labels` server-side, so a
    retry sees a smaller population and re-runs the line fit cleanly.
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
        flagged: list[LaserLabel] = []
        for i, is_outlier in enumerate(outlier_mask):
            if not is_outlier:
                continue
            label = positives[i]
            activity.logger.info(
                "dive_id=%d OUTLIER laser_label_id=%s image_id=%s "
                "x=%.1f y=%.1f perp=%.2fpx label_studio_task_id=%s "
                "label_studio_project_id=%s -> superseded=True",
                dive_id,
                label.id,
                label.image_id,
                float(label.x),
                float(label.y),
                float(perp[i]),
                label.label_studio_task_id,
                label.label_studio_project_id,
            )
            label.superseded = True
            flagged.append(label)

        # Concurrent supersede PUTs, capped by SUPERSEDE_CONCURRENCY.
        # `asyncio.gather` (return_exceptions=False) raises the first
        # exception bare — matches the existing failure-propagation
        # contract (TaskGroup would wrap in ExceptionGroup) — and lets
        # already-in-flight tasks run to completion, so partial
        # supersede progress survives a single failed PUT and the
        # next retry of the activity sees the cleaned subset.
        sem = asyncio.Semaphore(SUPERSEDE_CONCURRENCY)

        async def _supersede(label: LaserLabel) -> None:
            async with sem:
                await fs.labels.put_laser_label(label.image_id, label)
                activity.heartbeat()

        await asyncio.gather(*(_supersede(label) for label in flagged))

    activity.logger.info(
        "dive_id=%d superseded %d/%d positive laser labels",
        dive_id,
        n_outliers,
        xy.shape[0],
    )
    return n_outliers
