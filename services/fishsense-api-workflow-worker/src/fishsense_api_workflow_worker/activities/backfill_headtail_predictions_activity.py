"""Attach head/tail predictions to Label Studio tasks that already exist.

Populate seeds a task's pre-annotation exactly once, at import time, and
`import_tasks_and_record_labels` dedupes by URL — so once a dive's tasks exist,
a *new* `HeadTailPrediction` for one of those images changes the database and
nothing a labeler ever sees.

That is not an edge case here, it is most of the corpus: 39,766 head/tail tasks
already exist, of which 3,147 across 19 dives are still unlabelled. Without
this activity every one of them would keep its blank canvas no matter how good
the detector got, and the re-prediction cohort would be busywork.

The laser stage needed exactly this, and the slate detector hit the identical
gap first (#493, dive 65 seeded 0/28). LS lets a task carry several
predictions, so attaching one to an existing task is a `predictions.create`
rather than a re-import.

Idempotency keys on the LS `model_version` tag, which carries the stage version
— so a task seeded by an older stage does *not* look attached and gets the new
prediction, while a task already carrying the current version is skipped.
Re-running is free.

Completed tasks are skipped deliberately: a labeler has already placed those
points, and a fresh pre-annotation beside them would be noise at best and an
invitation to second-guess finished work at worst.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Set, Tuple

from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_headtail_label_studio_project_activity import (  # noqa: E501  pylint: disable=line-too-long
    prediction_annotations,
)
from fishsense_api_workflow_worker.activities.populate_utils import _get_ls_client
from fishsense_api_workflow_worker.activities.utils import get_fs_client


def select_attach_targets(predictions, headtail_labels) -> Dict[int, Tuple[int, int]]:
    """Map image_id -> (label_studio_task_id, label_studio_project_id).

    Eligible images have a prediction that would actually place something —
    reusing populate's own `prediction_annotations`, so the silhouette band and
    the abstention rules cannot drift between the two paths — and an
    *incomplete, non-superseded* LS task. First non-superseded task per image
    wins.
    """
    placeable: Set[int] = {
        p.image_id
        for p in predictions
        if p.image_id is not None and prediction_annotations(p)
    }
    targets: Dict[int, Tuple[int, int]] = {}
    for label in headtail_labels:
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


async def _attached_task_versions(ls, project_ids: Set[int]) -> Set[Tuple[int, str]]:
    """`(task_id, model_version)` pairs already attached, across these projects.

    Listed **once per project**, not per task: the corpus has 3,147 attachable
    tasks across 19 dives, and a per-task check would mean thousands of calls
    against hosted Label Studio for a single backfill run.

    Keyed on the version as well as the task because LS allows several
    predictions per task. A task seeded by an older stage must still receive
    the new one, while a task already carrying the current version is skipped —
    which is what makes re-running free.
    """
    attached: Set[Tuple[int, str]] = set()
    for project_id in project_ids:
        existing = await asyncio.to_thread(
            lambda pid=project_id: ls.predictions.list(project=pid)
        )
        for prediction in existing or []:
            attached.add((prediction.task, getattr(prediction, "model_version", None)))
    return attached


@activity.defn
async def backfill_headtail_predictions_for_dive_activity(dive_id: int) -> int:
    """Attach current-version `HeadTailPrediction`s to existing head/tail LS
    tasks. Returns the number newly attached (idempotent)."""
    async with get_fs_client() as fs:
        predictions = await fs.labels.get_headtail_predictions(dive_id) or []
        headtail_labels = await fs.labels.get_headtail_labels(dive_id) or []

        targets = select_attach_targets(predictions, headtail_labels)
        if not targets:
            activity.logger.info(
                "dive %d: no placeable head/tail predictions with attachable LS tasks",
                dive_id,
            )
            return 0

        prediction_by_image = {p.image_id: p for p in predictions}
        ls = _get_ls_client()
        project_ids = {project_id for _task_id, project_id in targets.values()}

        already = await _attached_task_versions(ls, project_ids)

        attached = 0
        for image_id, (task_id, _project_id) in targets.items():
            wrapper = prediction_annotations(prediction_by_image[image_id])
            if not wrapper:
                continue
            body = wrapper[0]
            if (task_id, body["model_version"]) in already:
                continue
            await asyncio.to_thread(
                lambda tid=task_id, payload=body: ls.predictions.create(
                    task=tid,
                    model_version=payload["model_version"],
                    result=payload["result"],
                )
            )
            attached += 1

    activity.logger.info(
        "dive %d: attached %d head/tail prediction(s) to existing tasks",
        dive_id,
        attached,
    )
    return attached
