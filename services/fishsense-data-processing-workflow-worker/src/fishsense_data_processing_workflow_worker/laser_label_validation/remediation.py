"""Plan the revival of laser labels the eroding validator superseded.

Pure: given a dive's labels (superseded included) and its calibration frames,
say which superseded labels to revive. The dry run, the apply step's safety
check and the tests all go through `plan_dive`, so there is one definition of
"what remediation would do". See `test_laser_supersede_remediation_plan.py`
for the constraints and why each exists.

The judgement is `judge_dive` — the validator's own — so a revived label is
exactly one the next hourly validator run keeps.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Iterable, List

from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_shared.laser_remediation import revival_digest

from fishsense_data_processing_workflow_worker.laser_label_validation.judgement import (
    judge_dive,
)

__all__ = ["EXCLUDED", "DivePlan", "plan_dive", "plan_digest"]

EXCLUDED = "excluded"


@dataclass
class DivePlan:  # pylint: disable=too-many-instance-attributes
    """One dive's row of the remediation report."""

    dive_id: int
    status: str
    positives: int
    superseded_now: int
    superseded_after: int
    revive_ids: List[int] = field(default_factory=list)
    # Superseded labels the fit keeps but the operator excluded.
    excluded_kept: List[int] = field(default_factory=list)
    # Superseded labels on a dive the fit could not judge; left alone.
    unjudged_superseded: int = 0
    reflection_suspect: dict | None = None
    # Revivals a human must follow up downstream (see module docs).
    revive_on_calibration_frames: List[int] = field(default_factory=list)
    revive_on_images_with_another_live_label: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """A JSON-ready report row."""
        return asdict(self)


def _is_superseded(label: LaserLabel) -> bool:
    # `is True`: a legacy NULL is neither live nor something this tool revives.
    return label.superseded is True


def plan_dive(  # pylint: disable=too-many-arguments
    dive_id: int,
    labels: Iterable[LaserLabel],
    calibration_image_ids: set,
    *,
    excluded_label_ids: set | None = None,
    dive_excluded: bool = False,
) -> DivePlan:
    """What remediation would do to one dive."""
    judgement = judge_dive(list(labels), calibration_image_ids)
    positives = judgement.positives
    superseded = [label for label in positives if _is_superseded(label)]
    plan = DivePlan(
        dive_id=dive_id,
        status=EXCLUDED if dive_excluded else judgement.status,
        positives=len(positives),
        superseded_now=len(superseded),
        superseded_after=len(superseded),
    )
    if judgement.reflection is not None:
        plan.reflection_suspect = {
            key: float(value) if isinstance(value, float) else value
            for key, value in asdict(judgement.reflection).items()
        }
    if dive_excluded:
        return plan
    if not judgement.judged:
        plan.unjudged_superseded = len(superseded)
        return plan

    excluded_label_ids = excluded_label_ids or set()
    kept = [
        label
        for label in superseded
        if label.id not in judgement.flagged_ids and label.completed is True
    ]
    plan.excluded_kept = sorted(
        label.id for label in kept if label.id in excluded_label_ids
    )
    revive = sorted(
        (label for label in kept if label.id not in excluded_label_ids),
        key=lambda label: label.id,
    )
    plan.revive_ids = [label.id for label in revive]
    plan.superseded_after = len(superseded) - len(revive)

    live_per_image = Counter(
        label.image_id for label in positives if label.superseded is False
    )
    plan.revive_on_calibration_frames = [
        label.id for label in revive if judgement.is_calibration(label.id)
    ]
    plan.revive_on_images_with_another_live_label = [
        label.id for label in revive if live_per_image[label.image_id] > 0
    ]
    return plan


def plan_digest(plans: Iterable[DivePlan]) -> str:
    """sha256 over exactly the revivals, independent of dive order.

    The apply step recomputes the plan and refuses unless this matches the
    reviewed report, so nothing is written that was not in front of a human.
    """
    return revival_digest((plan.dive_id, plan.revive_ids) for plan in plans)
