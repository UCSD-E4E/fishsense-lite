"""The per-dive laser-label judgement: which labels one line fit flags.

The single definition of what the validator decides, shared by the two things
that act on it — the hourly validator (supersedes what is flagged) and the
remediation tool (revives superseded labels the same judgement keeps). A
second copy would drift, and then a revived label is superseded again within
the hour.

It judges the dive's FULL population, superseded labels included, in
(image_id, id) order — the contract fishsense-core #88 documents: re-judging
survivors erodes a dive a pass at a time, and RANSAC's answer depends on row
order. See `test_laser_validator_does_not_erode.py`.

`judged` separates a verdict from an abstention. Too few positives, a line
that is not confident, a reflection split and a refused (>50%) fit all flag
nothing, and none of them says the labels are good.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_core.laser import (
    MIN_POINTS_FOR_LINE,
    LineFit,
    fit_dive_line,
    flag_outliers,
)

from fishsense_data_processing_workflow_worker.laser_label_validation.reflection import (
    ReflectionSuspect,
    detect_reflection_split,
)

__all__ = [
    "MAX_OUTLIER_FRACTION",
    "DiveJudgement",
    "calibration_image_ids",
    "judge_dive",
]

# Safety gate: refuse to act when more than this fraction of a dive's positive
# labels would be flagged. At >50% the line fit is more likely degenerate (a
# small accidentally-aligned cluster picked over the real majority) than the
# labelers wrong at that rate. With core's signed-residual MAD this needs the
# fitted line to sit far from the bulk of the dots, which is exactly that case.
MAX_OUTLIER_FRACTION = 0.5

#: Verdicts. Only the last two are judgements.
TOO_FEW = "too_few"
NO_FIT = "no_fit"
NOT_CONFIDENT = "not_confident"
REFLECTION = "reflection"
GATE = "gate"
NO_OUTLIERS = "no_outliers"
FLAGGED = "flagged"


@dataclass
class DiveJudgement:  # pylint: disable=too-many-instance-attributes
    """What one fit made of a dive. Ids are `LaserLabel.id`s."""

    status: str
    positives: List[LaserLabel] = field(default_factory=list)
    fit: LineFit | None = None
    reflection: ReflectionSuspect | None = None
    flagged_ids: set = field(default_factory=set)
    n_flagged_before_gate: int = 0
    perpendicular_px: dict = field(default_factory=dict)
    calibration_ids: set = field(default_factory=set)

    @property
    def judged(self) -> bool:
        """True only for a confident, un-refused fit — a verdict, not an
        abstention."""
        return self.status in (NO_OUTLIERS, FLAGGED)

    @property
    def kept_ids(self) -> set:
        """Positives the judgement did not flag. Meaningful only if judged."""
        return {label.id for label in self.positives} - self.flagged_ids

    def is_calibration(self, label_id) -> bool:
        """Whether that label sits on a calibration (slate) frame."""
        return label_id in self.calibration_ids


def calibration_image_ids(slate_labels: Iterable) -> set:
    """Frames that are calibration observations: a completed, non-superseded
    `DiveSlateLabel`, exactly what stage 13 consumes."""
    return {
        label.image_id
        for label in slate_labels or []
        if label.image_id is not None
        and label.completed
        and not getattr(label, "superseded", False)
    }


def _positives(labels: Iterable[LaserLabel]) -> List[LaserLabel]:
    """Labels with both coordinates, in (image_id, id) order.

    Sentinel rows seeded by populate (no laser visible) and skipped
    annotations carry null x/y and are not part of the population. The order is
    imposed here rather than trusted from the API: RANSAC picks point pairs by
    row index, so the same labels in another order can settle on another line
    (fishsense-core measured one dive flagging 41-63 labels across shuffles).
    """
    return sorted(
        (label for label in labels if label.x is not None and label.y is not None),
        key=lambda label: (label.image_id, label.id),
    )


def judge_dive(
    labels: Iterable[LaserLabel], calibration_image_ids: set
) -> DiveJudgement:
    # pylint: disable=too-many-return-statements
    #   One early return per verdict, in the order they are decided; the
    #   validator's own body had the same shape before it moved here.
    """Fit one line through the dive's positives and flag its outliers.

    `calibration_image_ids` are frames carrying a completed, non-superseded
    `DiveSlateLabel`; they are judged against the coarse absolute bound rather
    than 3 sigma (see `test_coarse_calibration_frame_supersede.py`).
    """
    positives = _positives(labels)
    calibration_ids = {
        label.id for label in positives if label.image_id in calibration_image_ids
    }
    judgement = DiveJudgement(
        status=TOO_FEW, positives=positives, calibration_ids=calibration_ids
    )
    if len(positives) < MIN_POINTS_FOR_LINE:
        return judgement

    xy = np.array([(float(label.x), float(label.y)) for label in positives])
    judgement.fit = fit_dive_line(xy)
    if judgement.fit is None:
        judgement.status = NO_FIT
        return judgement

    # Before confidence: a near-even split between a laser and its reflection
    # is exactly what collapses confidence, and it needs naming, not silence.
    judgement.reflection = detect_reflection_split(xy, judgement.fit)
    if judgement.reflection is not None:
        judgement.status = REFLECTION
        return judgement

    if not judgement.fit.is_confident:
        judgement.status = NOT_CONFIDENT
        return judgement

    mask = np.array([label.id in calibration_ids for label in positives])
    flags = flag_outliers(xy, judgement.fit, calibration_mask=mask)
    judgement.n_flagged_before_gate = int(flags.sum())
    perp = judgement.fit.perpendicular_distance(xy[:, 0], xy[:, 1])
    judgement.perpendicular_px = {
        label.id: float(d) for label, d in zip(positives, perp)
    }
    if judgement.n_flagged_before_gate == 0:
        judgement.status = NO_OUTLIERS
        return judgement
    if judgement.n_flagged_before_gate / len(positives) > MAX_OUTLIER_FRACTION:
        judgement.status = GATE
        return judgement

    judgement.status = FLAGGED
    judgement.flagged_ids = {
        label.id for label, flagged in zip(positives, flags) if flagged
    }
    return judgement
