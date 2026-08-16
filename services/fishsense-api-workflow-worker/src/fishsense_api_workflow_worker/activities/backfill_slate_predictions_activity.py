"""Attach persisted `SlatePrediction` rows to EXISTING dive-slate LS tasks.

**RETIRED 2026-08-03 — registered, but nothing schedules it.** The
ECC >= 0.80 acceptance gate does not transfer out of distribution: pool
dives produced high-ECC (0.93-0.97) *false* fits that sailed through it
(prod dives 65/71/77/80/83, all pool). The team declined an
active-learning loop; `predict-slate-images-workflow-schedule` is now
actively deleted at worker startup (`worker._RETIRED_SCHEDULE_IDS`) and
the 130 seeded Label Studio predictions were removed.

The code is kept registered so a future evaluation can start it by hand
— it is dormant, not dead — but nothing invokes it on its own. Do not
read it as part of the live pipeline.


The dive-slate populate seeds model pre-annotations only at *import* time and
runs exactly once per dive (stage-9-cohort gated: a slate frame with no
`DiveSlateLabel` row). So a dive whose LS tasks were imported before the slate
predictor existed — or before its predictions landed — never receives them; the
predictions sit in the DB unused, and the dive has already dropped out of every
cohort that could re-import.

This activity closes that gap by POSTing the predictions to the already-imported
LS tasks via the LS predictions API (no task re-import, so no duplicate tasks).
It is idempotent — a task that already carries a `slate-detector` prediction is
skipped — so it is safe to run repeatedly (the predict parent calls it after
every persist) and on demand (``BackfillSlatePredictionsWorkflow``) to catch up
dives predicted before this shipped.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Set, Tuple

from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
from fishsense_api_sdk.models.slate_prediction import SlatePrediction
from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_dive_slate_label_studio_project_activity import (  # noqa: E501  pylint: disable=line-too-long
    SLATE_DETECTOR_MODEL_VERSION,
    _prediction_annotations,
    _slate_panel_aspect,
)
from fishsense_api_workflow_worker.activities.populate_utils import _get_ls_client
from fishsense_api_workflow_worker.activities.sync_dive_slate_labels_for_label_studio_project_activity import (  # noqa: E501  pylint: disable=line-too-long
    compute_pdf_panel_width_in_composite,
)
from fishsense_api_workflow_worker.activities.utils import get_fs_client


def _select_attach_targets(
    predictions: List[SlatePrediction],
    slate_labels: List[DiveSlateLabel],
) -> Dict[int, Tuple[int, int]]:
    """Map image_id -> (label_studio_task_id, label_studio_project_id).

    Only images that have a *seeded* prediction (non-empty `reference_points`)
    AND an *incomplete, non-superseded* LS task are eligible. Completed tasks are
    skipped (a labeler already placed the points; a pre-annotation would be
    noise); declined predictions seed nothing. First non-superseded task per
    image wins.
    """
    seeded: Set[int] = {
        p.image_id
        for p in predictions
        if p.image_id is not None and p.reference_points
    }
    targets: Dict[int, Tuple[int, int]] = {}
    for label in slate_labels:
        if label.completed or label.superseded:
            continue
        if label.label_studio_task_id is None or label.label_studio_project_id is None:
            continue
        if label.image_id not in seeded or label.image_id in targets:
            continue
        targets[label.image_id] = (
            int(label.label_studio_task_id),
            int(label.label_studio_project_id),
        )
    return targets


async def _tasks_with_existing_slate_prediction(ls, project_ids: Set[int]) -> Set[int]:
    """Task ids in `project_ids` that already carry a slate-detector prediction.

    Listed once per project (not per task) so the idempotency check is a handful
    of calls regardless of task count. LS allows multiple predictions per task,
    so without this the backfill would stack duplicates on every re-run.
    """
    already: Set[int] = set()
    for project_id in project_ids:
        existing = await asyncio.to_thread(
            lambda pid=project_id: ls.predictions.list(project=pid)
        )
        for prediction in existing or []:
            if getattr(prediction, "model_version", None) == SLATE_DETECTOR_MODEL_VERSION:
                already.add(prediction.task)
    return already


@activity.defn
async def backfill_slate_predictions_for_dive_activity(dive_id: int) -> int:
    """Attach seeded `SlatePrediction`s to existing dive-slate LS tasks.

    Returns the number of predictions newly attached (idempotent — already-
    attached tasks are skipped).
    """
    async with get_fs_client() as fs:
        predictions = await fs.labels.get_slate_predictions(dive_id) or []
        slate_labels = await fs.labels.get_dive_slate_labels(dive_id) or []

        targets = _select_attach_targets(predictions, slate_labels)
        if not targets:
            activity.logger.info(
                "dive %d: no seeded predictions with attachable LS tasks", dive_id
            )
            return 0

        prediction_by_image = {p.image_id: p for p in predictions}
        # Panel width converts photo-space points to the LS composite canvas
        # (inverse of the sync's panel-offset strip); None -> no shift.
        aspect = await _slate_panel_aspect(dive_id, fs)

        ls = _get_ls_client()
        project_ids = {project_id for _task_id, project_id in targets.values()}
        already = await _tasks_with_existing_slate_prediction(ls, project_ids)

        attached = 0
        for image_id, (task_id, _project_id) in targets.items():
            if task_id in already:
                continue
            prediction = prediction_by_image[image_id]
            panel_width = (
                compute_pdf_panel_width_in_composite(aspect, prediction.height)
                if aspect is not None and prediction.height
                else 0.0
            )
            wrapper = _prediction_annotations(prediction, panel_width)
            if not wrapper:
                continue
            body = wrapper[0]
            await asyncio.to_thread(
                lambda tid=task_id, payload=body: ls.predictions.create(
                    task=tid,
                    model_version=payload["model_version"],
                    score=payload["score"],
                    result=payload["result"],
                )
            )
            attached += 1
            activity.heartbeat()

        activity.logger.info(
            "dive %d: attached %d slate predictions to existing LS tasks",
            dive_id,
            attached,
        )
        return attached
