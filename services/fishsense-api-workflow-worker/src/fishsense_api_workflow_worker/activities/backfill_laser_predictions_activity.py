"""Attach laser predictions to Label Studio tasks that already exist.

Populate seeds a task's pre-annotation exactly once, at import time, and
`import_tasks_and_record_labels` dedupes by URL — so once a dive's tasks exist,
a *new* `LaserPrediction` for one of those images changes the database and
nothing a labeler ever sees. That is the whole population re-prediction
targets: a dive is only eligible for re-prediction while it is still being
labeled, which is exactly when its tasks are already open.

Without this activity the re-prediction cohort would be busywork. The slate
detector hit the identical gap and fixed it the same way (#493): LS lets a task
carry several predictions, so attaching one to an existing task is a
`predictions.create` rather than a re-import.

Idempotency keys on the LS `model_version` tag, which now carries the stage
version (`laser-detector-v2`) instead of the bare constant it used to be. So a
task seeded by an older stage does *not* look attached, and gets the new
prediction; a task already carrying the current version is skipped. Re-running
is free.

Completed tasks are skipped deliberately: a labeler has already placed that
point, and a fresh pre-annotation beside it would be noise at best and an
invitation to second-guess finished work at worst.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Set, Tuple

from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_sdk.models.laser_prediction import LaserPrediction
from fishsense_shared import laser_model_version_tag
from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_laser_label_studio_project_activity import (  # noqa: E501  pylint: disable=line-too-long
    _prediction_annotations,
    dive_laser_label,
)
from fishsense_api_workflow_worker.activities.populate_utils import _get_ls_client
from fishsense_api_workflow_worker.activities.utils import get_fs_client


def _select_attach_targets(
    predictions: List[LaserPrediction],
    laser_labels: List[LaserLabel],
) -> Dict[int, Tuple[int, int]]:
    """Map image_id -> (label_studio_task_id, label_studio_project_id).

    Eligible images have a prediction carrying a dot (x/y set — a non-detection
    or a point rejected for falling outside the expected-laser region seeds
    nothing) and an *incomplete, non-superseded* LS task. First non-superseded
    task per image wins.
    """
    placeable: Set[int] = {
        prediction.image_id
        for prediction in predictions
        if prediction.image_id is not None
        and prediction.x is not None
        and prediction.y is not None
    }
    targets: Dict[int, Tuple[int, int]] = {}
    for label in laser_labels:
        if label.completed or label.superseded:
            continue
        if label.label_studio_task_id is None or label.label_studio_project_id is None:
            continue
        if label.image_id not in placeable or label.image_id in targets:
            continue
        targets[label.image_id] = (
            int(label.label_studio_task_id),
            int(label.label_studio_project_id),
        )
    return targets


async def _tasks_with_current_laser_prediction(ls, project_ids: Set[int]) -> Set[int]:
    """Task ids already carrying a prediction from the *current* stage version.

    Listed once per project rather than per task, so the check costs a handful
    of calls regardless of task count. LS allows several predictions per task,
    so without this a re-run would stack duplicates on every firing.
    """
    tag = laser_model_version_tag()
    already: Set[int] = set()
    for project_id in project_ids:
        existing = await asyncio.to_thread(
            lambda pid=project_id: ls.predictions.list(project=pid)
        )
        for prediction in existing or []:
            if getattr(prediction, "model_version", None) == tag:
                already.add(prediction.task)
    return already


@activity.defn
async def backfill_laser_predictions_for_dive_activity(dive_id: int) -> int:
    """Attach current-version `LaserPrediction`s to existing laser LS tasks.

    Returns the number newly attached (idempotent — tasks already carrying the
    current version are skipped).
    """
    async with get_fs_client() as fs:
        predictions = await fs.labels.get_laser_predictions(dive_id) or []
        laser_labels = await fs.labels.get_laser_labels(dive_id) or []

        targets = _select_attach_targets(predictions, laser_labels)
        if not targets:
            activity.logger.info(
                "dive %d: no placeable predictions with attachable LS tasks", dive_id
            )
            return 0

        prediction_by_image = {p.image_id: p for p in predictions}
        # One colour for the whole dive, same rule populate uses, so a
        # backfilled task cannot disagree with its neighbours.
        laser_label = dive_laser_label(predictions)

        ls = _get_ls_client()
        project_ids = {project_id for _task_id, project_id in targets.values()}
        already = await _tasks_with_current_laser_prediction(ls, project_ids)

        attached = 0
        for image_id, (task_id, _project_id) in targets.items():
            if task_id in already:
                continue
            wrapper = _prediction_annotations(
                prediction_by_image[image_id], laser_label
            )
            if not wrapper:
                continue
            body = wrapper[0]
            await asyncio.to_thread(
                lambda tid=task_id, payload=body: ls.predictions.create(
                    task=tid,
                    model_version=payload["model_version"],
                    result=payload["result"],
                )
            )
            attached += 1
            activity.heartbeat()

        activity.logger.info(
            "dive %d: attached %d laser predictions to existing LS tasks (%s)",
            dive_id,
            attached,
            laser_label,
        )
        return attached
