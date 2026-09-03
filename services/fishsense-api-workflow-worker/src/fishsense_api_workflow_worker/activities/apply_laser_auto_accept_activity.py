"""Apply auto-accept verdicts to Label Studio tasks that already exist.

Populate imports an auto-accepted frame already annotated, but it imports a
task exactly once — so for a dive whose tasks already exist, the gate's verdict
changes the database and nothing a labeler sees. That is *every* dive still
being labeled, which is exactly the population auto-accept exists to relieve.
Without this activity the feature would only ever help dives populated after it
shipped.

The slate detector hit the identical gap (#493) and the laser pre-annotations
hit it after that; both fixed it the same way, by attaching to the open task
rather than re-importing it. This is the third instance, and the shape is the
same: list once per project, skip what is already done, attach the rest.

**The safety rule here is stricter than for a pre-annotation, and deliberately
so.** A pre-annotation sits beside a labeler's work as a suggestion; an
annotation is a claim that the work is finished. So this touches a task only
when Label Studio itself says nobody has started on it — no annotation and no
draft. That single condition covers three separate hazards at once:

* **Idempotency.** The first pass leaves an annotation behind, so the second
  pass sees a task that is no longer eligible. No version tag needed, unlike
  the prediction backfill.
* **Never overwriting a human.** A labeler who completed the task between the
  gate running and this activity firing is protected, even though the DB row
  this activity read still said incomplete.
* **Never discarding work in progress.** A draft means someone is mid-click on
  that exact frame. Nothing else here would notice: the DB says incomplete and
  LS says un-annotated, both true, and annotating underneath them would throw
  away what they are doing.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Set, Tuple

from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_sdk.models.laser_prediction import LaserPrediction
from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_laser_label_studio_project_activity import (  # noqa: E501  pylint: disable=line-too-long
    _auto_accepted_annotations,
    dive_laser_label,
)
from fishsense_api_workflow_worker.activities.populate_utils import _get_ls_client
from fishsense_api_workflow_worker.activities.utils import get_fs_client

__all__ = ["apply_laser_auto_accept_for_dive_activity"]


def _select_auto_accept_targets(
    predictions: List[LaserPrediction],
    laser_labels: List[LaserLabel],
) -> Dict[int, Tuple[int, int]]:
    """Map image_id -> (label_studio_task_id, label_studio_project_id).

    Eligible images have a prediction the gate cleared *and* that carries a
    placeable dot, plus an incomplete, non-superseded LS task. First
    non-superseded task per image wins.

    The `x`/`y` check is belt-and-braces: the gate returns `no_prediction` for
    an abstention and never auto-accepts one, so a cleared prediction without a
    dot should not exist. If one ever does, there is nothing to annotate with
    and silently skipping beats writing a keypoint at a missing coordinate.
    """
    cleared: Set[int] = {
        prediction.image_id
        for prediction in predictions
        if prediction.image_id is not None
        and getattr(prediction, "auto_accept", False)
        and prediction.x is not None
        and prediction.y is not None
    }
    targets: Dict[int, Tuple[int, int]] = {}
    for label in laser_labels:
        if label.completed or label.superseded:
            continue
        if label.label_studio_task_id is None or label.label_studio_project_id is None:
            continue
        if label.image_id not in cleared or label.image_id in targets:
            continue
        targets[label.image_id] = (
            int(label.label_studio_task_id),
            int(label.label_studio_project_id),
        )
    return targets


async def _untouched_task_ids(ls, project_ids: Set[int]) -> Set[int]:
    """Task ids that Label Studio says nobody has started: no annotation and
    no draft.

    Listed once per project rather than fetched per task, so the check costs a
    handful of calls regardless of how many frames a dive has.

    Note this is an allow-list, not a deny-list, and that matters for a task
    the listing does not mention at all — a project that was deleted, or tasks
    cleaned up by hand. Unknown stays unknown: annotating blind would 404, and
    raising would wedge every other frame in the dive behind one stale row.
    """
    untouched: Set[int] = set()
    for project_id in project_ids:
        tasks = await asyncio.to_thread(
            lambda pid=project_id: ls.tasks.list(project=pid)
        )
        for task in tasks or []:
            if getattr(task, "annotations", None):
                continue
            if getattr(task, "drafts", None):
                continue
            untouched.add(task.id)
    return untouched


@activity.defn
async def apply_laser_auto_accept_for_dive_activity(dive_id: int) -> int:
    """Annotate `dive_id`'s open LS tasks whose predictions the gate cleared.

    Returns the number of annotations created. Idempotent: a task annotated by
    an earlier run is no longer untouched, so it is skipped.

    The `LaserLabel` row is deliberately not written here. The hourly sync
    stays the single writer of label x/y and `completed`, reading them back out
    of LS exactly as it does for a human annotation — so an auto-accepted frame
    reaches the database by the same path as every other label, and there is no
    second writer to disagree with.
    """
    async with get_fs_client() as fs:
        predictions = await fs.labels.get_laser_predictions(dive_id) or []
        laser_labels = await fs.labels.get_laser_labels(dive_id) or []

        targets = _select_auto_accept_targets(predictions, laser_labels)
        if not targets:
            activity.logger.info(
                "dive %d: no auto-accepted predictions with attachable LS tasks",
                dive_id,
            )
            return 0

        prediction_by_image = {p.image_id: p for p in predictions}
        # One colour for the whole dive, the same rule populate uses, so an
        # annotation attached here cannot disagree with its neighbours.
        laser_label = dive_laser_label(predictions)

        ls = _get_ls_client()
        project_ids = {project_id for _task_id, project_id in targets.values()}
        untouched = await _untouched_task_ids(ls, project_ids)

        applied = 0
        for image_id, (task_id, project_id) in targets.items():
            if task_id not in untouched:
                continue
            wrapper = _auto_accepted_annotations(
                prediction_by_image[image_id], laser_label
            )
            if not wrapper:
                continue
            await asyncio.to_thread(
                lambda tid=task_id, pid=project_id, payload=wrapper[0]: (
                    ls.annotations.create(id=tid, project=pid, result=payload["result"])
                )
            )
            applied += 1

        activity.logger.info(
            "dive %d: auto-accepted %d/%d open tasks (%d were already started)",
            dive_id,
            applied,
            len(targets),
            len(targets) - applied,
        )
        return applied
