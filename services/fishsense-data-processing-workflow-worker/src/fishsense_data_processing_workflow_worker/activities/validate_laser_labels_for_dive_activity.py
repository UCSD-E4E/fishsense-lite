"""Per-dive laser-label validation.

Fetches the dive's `LaserLabel`s from fishsense-api (superseded included),
fits a RANSAC line through the positives, and supersedes any label
whose perpendicular distance exceeds the per-dive outlier threshold.
The laser rig is fixed across a dive, so all positive laser
observations should be colinear in image space — outliers are
mislabeled and shouldn't feed downstream stage 13 calibration / stage
14 measurement.

Calibration frames are judged coarsely. A frame carrying a completed,
non-superseded `DiveSlateLabel` is a calibration observation, and the dive
line is fitted overwhelmingly from the *measurement* frames — so a genuine
slate dot sits a few px off it (measured on prod: 0.34-8.48 px on dive 347,
against a fish-dot line of 1.25 px median) and 3 sigma superseded it. That is
how 347 came to be calibrated from ONE frame and 349 from two dots 26 cm
apart in range, and it cannot be undone by relabelling. Those frames are
therefore tested against `COARSE_CALIBRATION_TOLERANCE_PX` instead, which
still removes the separate population of wild mislabels (45-130 px on 347).

Each run is ONE judgement of the dive's FULL population. The fetch includes
superseded labels, the rows are put in (image_id, id) order, and the fit and
flag run over all of them; a flagged label that is still live is superseded.
Nothing is carried over from earlier runs, so an unchanged dive gets the same
answer every hour.

It used to re-fit only the survivors, and that eroded dives a pass at a time:
`flag_outliers` estimates its noise scale from the rows it is handed, removing
the flagged tail narrows that population, and the next estimate flags more.
Prod dive 521 lost 15, then 7, then 1 label on consecutive hourly runs with no
label edited in between. fishsense-core #88 documents the contract this
follows (fit and flag the full population every run, in a stable order); see
`test_laser_validator_does_not_erode.py`.

The validator only supersedes. A superseded label the fit now keeps — most
of what the erosion took — is left alone: reviving is a reviewed operator
decision, not something an hourly job does. A legacy row whose `superseded`
is NULL is fitted but never written; every resolver already reads it as not
live.

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
from fishsense_api_sdk.models.dive_laser_line import DiveLaserLine
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_core.laser import (
    COARSE_CALIBRATION_TOLERANCE_PX,
    MIN_POINTS_FOR_LINE,
    fit_dive_line,
    flag_outliers,
)
from temporalio import activity

from fishsense_data_processing_workflow_worker.activities.heartbeat import (
    HEARTBEAT_INTERVAL_SECONDS,
    heartbeat_pump,
)
from fishsense_data_processing_workflow_worker.activities.utils import get_fs_client
from fishsense_data_processing_workflow_worker.laser_label_validation.reflection import (
    detect_reflection_split,
)

__all__ = [
    "validate_laser_labels_for_dive_activity",
    "SUPERSEDE_CONCURRENCY",
    "HEARTBEAT_INTERVAL_SECONDS",
    "MAX_OUTLIER_FRACTION",
]

# Bound on concurrent supersede PUTs per dive. A dive with many flagged
# outliers was blowing `start_to_close` (10m) on sequential PUTs because
# the SDK's 10s-timeout × 3-retry ladder per PUT compounds linearly. With
# a cap of 8, the budget is N/8 × per-PUT cost — comfortable for any
# realistic outlier count. The cap also keeps a single dive from hogging
# every outbound HTTP slot when multiple validate workflows run in
# parallel against different dives.
SUPERSEDE_CONCURRENCY = 8

# Safety gate: refuse to supersede when more than this fraction of a
# dive's positive labels would be flagged. At >50% the line fit is
# more likely to be degenerate (a small accidentally-aligned cluster
# being picked over the real majority) than the labelers being wrong
# at that rate. Empirically prod has ~6 dives at 50%+ supersede rate
# that are almost certainly degenerate fits — refusing to act on them
# costs us nothing (the labels stay as-is, available for manual
# review) and prevents propagating the line-fit error to the DB.
MAX_OUTLIER_FRACTION = 0.5


def _positive_xy(labels: List[LaserLabel]) -> tuple[np.ndarray, List[LaserLabel]]:
    """Pull the positive (x, y) labels and the matching label objects.

    A "positive" is a laser-localization label with both coordinates set.
    Sentinel rows seeded by populate (no laser visible) and skipped
    annotations land here as null x/y and are excluded.

    Ordered by (image_id, id) here rather than trusting the API: RANSAC picks
    point pairs by row index, so the same labels in another order can settle
    on another line (fishsense-core measured dive 257 flagging 41-63 labels
    across 50 shuffles).
    """
    positives = sorted(
        (label for label in labels if label.x is not None and label.y is not None),
        key=lambda label: (label.image_id, label.id),
    )
    if not positives:
        return np.empty((0, 2), dtype=float), []
    xy = np.array(
        [(float(label.x), float(label.y)) for label in positives], dtype=float
    )
    return xy, positives


@activity.defn
async def validate_laser_labels_for_dive_activity(dive_id: int) -> int:
    # pylint: disable=too-many-return-statements
    # Flat guard -> log -> return-0 per skip reason (no labels, too few
    # positives, no fit, reflection split, no outliers, fraction gate);
    # reads better inline than dispersed across helpers.
    """Run RANSAC line-fit validation for `dive_id` and supersede any
    flagged outliers that are still live. Returns the number of labels
    this run superseded (0 when the line isn't confident, there aren't
    enough positives to fit, or everything flagged is already superseded).

    Heartbeats around the SDK fetch + each compute milestone + each
    supersede write so a stalled call (large dive's `label_studio_json`
    payload over Traefik, slow PUT, etc.) trips `heartbeat_timeout`
    instead of grinding to `schedule_to_close_timeout` with no signal
    of where it hung.

    Failure semantics: if any individual `put_laser_label` raises, the
    activity raises and Temporal retries the whole activity. The retry
    re-judges the same full population, flags the same set, and writes
    only what the failed attempt had not yet superseded.
    """
    activity.logger.info(
        "dive_id=%d validation starting; fetching laser labels", dive_id
    )
    activity.heartbeat()
    async with heartbeat_pump(HEARTBEAT_INTERVAL_SECONDS), get_fs_client() as fs:
        labels = (
            await fs.labels.get_laser_labels(dive_id, include_superseded=True) or []
        )
        activity.logger.info(
            "dive_id=%d fetched %d laser label rows", dive_id, len(labels)
        )
        activity.heartbeat()

        if not labels:
            activity.logger.info(
                "dive_id=%d has no laser labels; skipping validation", dive_id
            )
            return 0

        # Which frames are calibration observations. Fetched before the fit
        # so a transport failure fails the activity (Temporal retries it)
        # rather than silently falling through to the tight rule and
        # superseding the dive's slate dots -- the failure this exists to
        # prevent. `stage 13` consumes only completed, non-superseded slate
        # labels, so those are exactly the frames that get the loose bound.
        slate_labels = await fs.labels.get_dive_slate_labels(dive_id) or []
        calibration_image_ids = {
            label.image_id
            for label in slate_labels
            if label.image_id is not None
            and label.completed
            and not getattr(label, "superseded", False)
        }
        activity.logger.info(
            "dive_id=%d has %d calibration frames among its slate labels; "
            "those are judged at %.0fpx rather than 3 sigma",
            dive_id,
            len(calibration_image_ids),
            COARSE_CALIBRATION_TOLERANCE_PX,
        )
        activity.heartbeat()

        xy, positives = _positive_xy(labels)
        if xy.shape[0] < MIN_POINTS_FOR_LINE:
            activity.logger.info(
                "dive_id=%d has %d positive laser labels (<%d); skipping line fit",
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

        # Persist the line fingerprint (byproduct we already computed) so
        # (camera_id, line) becomes queryable: borrow candidates, drift,
        # mount-swap epochs, pooled calibration. Written every run — even a
        # clean dive with no outliers — and tightens as outliers are
        # superseded across runs. Upsert keyed on dive_id.
        await fs.dives.put_dive_laser_line(
            dive_id,
            DiveLaserLine(
                dive_id=dive_id,
                a=fit.a,
                b=fit.b,
                c=fit.c,
                n_points=fit.n_points,
                inlier_count=fit.inlier_count,
                inlier_fraction=fit.inlier_fraction,
                residual_std=fit.residual_std,
                label_noise_mad=fit.label_noise_mad,
                line_confidence=fit.line_confidence,
            ),
        )
        activity.heartbeat()

        # Two-line (specular reflection) detection. A dive whose dots split
        # across two coherent parallel lines — labelers clicking the laser's
        # reflection on the slate (prod dive 77) — defeats single-line
        # validation in exactly the wrong way: with the artifact in the
        # majority, RANSAC anchors on the WRONG line and 3-sigma flagging
        # would supersede the true one; with a near-even split, confidence
        # collapses and the dive is silently skipped while stage 13 consumes
        # the poisoned mix. Choosing which line is real needs cross-dive
        # consensus (sibling dives, same camera), which the per-dive validator
        # doesn't have — so on detection it logs loudly and stands down,
        # leaving the labels for operator remediation (supersede the artifact
        # line, delete the extrinsics, refit — see the dive-77 recipe).
        suspect = detect_reflection_split(xy, fit)
        if suspect is not None:
            activity.logger.error(
                "dive_id=%d REFLECTION SUSPECT: laser dots form two parallel "
                "lines — primary n=%d, secondary n=%d at %.1fpx separation "
                "(angle %.2f deg). Likely specular-reflection mislabels; "
                "single-line validation cannot resolve which is real. "
                "Skipping supersede; manual remediation required before this "
                "dive's labels feed stage 13.",
                dive_id,
                suspect.n_primary,
                suspect.n_secondary,
                suspect.separation_px,
                suspect.angle_deg,
            )
            return 0

        calibration_mask = np.array(
            [label.image_id in calibration_image_ids for label in positives],
            dtype=bool,
        )
        outlier_mask = flag_outliers(xy, fit, calibration_mask=calibration_mask)
        n_outliers = int(outlier_mask.sum())
        if n_outliers == 0:
            activity.logger.info("dive_id=%d: no outlier laser labels", dive_id)
            return 0

        outlier_fraction = n_outliers / xy.shape[0]
        if outlier_fraction > MAX_OUTLIER_FRACTION:
            activity.logger.warning(
                "dive_id=%d would supersede %d/%d positive laser labels "
                "(%.0f%%, gate=%.0f%%); refusing — line fit is likely "
                "degenerate. Labels left unchanged for manual review.",
                dive_id,
                n_outliers,
                xy.shape[0],
                100.0 * outlier_fraction,
                100.0 * MAX_OUTLIER_FRACTION,
            )
            return 0

        perp = fit.perpendicular_distance(xy[:, 0], xy[:, 1])
        flagged: list[LaserLabel] = []
        for i, is_outlier in enumerate(outlier_mask):
            # `is False`, not `not`: a legacy NULL row is not live either.
            if not is_outlier or positives[i].superseded is not False:
                continue
            label = positives[i]
            activity.logger.info(
                "dive_id=%d OUTLIER laser_label_id=%s image_id=%s "
                "x=%.1f y=%.1f perp=%.2fpx rule=%s label_studio_task_id=%s "
                "label_studio_project_id=%s -> superseded=True",
                dive_id,
                label.id,
                label.image_id,
                float(label.x),
                float(label.y),
                float(perp[i]),
                "coarse-calibration" if calibration_mask[i] else "3-sigma",
                label.label_studio_task_id,
                label.label_studio_project_id,
            )
            label.superseded = True
            flagged.append(label)

        if not flagged:
            activity.logger.info(
                "dive_id=%d: %d outlier laser labels, all already superseded",
                dive_id,
                n_outliers,
            )
            return 0

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
        "dive_id=%d superseded %d/%d positive laser labels (%d flagged in all)",
        dive_id,
        len(flagged),
        xy.shape[0],
        n_outliers,
    )
    return len(flagged)
