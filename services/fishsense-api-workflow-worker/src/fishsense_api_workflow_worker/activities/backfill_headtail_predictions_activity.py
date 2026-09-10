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
from collections import Counter
from typing import Dict, Set, Tuple

from fishsense_shared.headtail_predictor import headtail_model_version_tag
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


async def _ensure_projects_show_predictions(ls, tags_by_project) -> int:
    """Point each project's `model_version` at a tier it actually has.

    **Attaching a prediction is not enough to show one.** Label Studio
    surfaces predictions to annotators only for the version named in the
    *project's* `model_version`; with it unset, `show_collab_predictions=True`
    and hundreds of stored predictions still render a blank task. A labeler
    worked through five frames of dive 94 by hand on 2026-09-10 with 334
    invisible predictions sitting on the project.

    It went unnoticed because `import_tasks` sets the field for free when the
    tasks carry `predictions` inline -- so a dive whose tasks were created
    *after* its predictions existed looked fine, and only dives backfilled
    onto pre-existing tasks were blank. That is most of the corpus. The laser
    stage sets `laser-detector-v2` the same way and has always worked, which
    is why the difference never showed up as a laser bug.

    `tags_by_project` maps a project id to a count of the tags its placeable
    predictions carry. A project displays exactly one version, and head/tail
    predictions are two-tier, so the tier is chosen from what the project
    *has*: the current `HEADTAIL_PREDICTOR_VERSION` tag when any prediction
    carries it, otherwise the most common tag present.

    Preferring the current tier is what makes a mixed dive -- part GPU, part
    Mask R-CNN fallback -- show SAM 3.1, since fallback rows are queued for
    upgrade anyway (`HEADTAIL_FALLBACK_PREDICTOR_VERSION`). But it must be
    *preference*, not a constant: a dive predicted entirely on the fallback
    has no SAM 3.1 predictions at all, so pinning the constant would point the
    project at a version it does not have, leaving every task blank -- and
    would overwrite a working value LS had already set from inline fallback
    predictions, hiding predictions that were visible.

    **Reachability.** This runs only where the backfill does: once per dive,
    in the predict parent, right after the predictions are persisted. That
    covers every dive predicted from here on. It does *not* revisit a dive
    already at the current predictor version, because such a dive has left the
    stale cohort and the parent will not select it again -- so a project whose
    `model_version` is later cleared or changed is not self-healing. The
    on-demand `BackfillHeadtailPredictionsWorkflow(dive_id)` is the repair
    path for that. The 18 projects blank at the time of this fix were repaired
    directly rather than waiting for a re-prediction that would never come.
    """
    current = headtail_model_version_tag()
    updated = 0
    for project_id, tags in tags_by_project.items():
        if not tags:
            continue
        want = current if current in tags else tags.most_common(1)[0][0]
        try:
            project = await asyncio.to_thread(ls.projects.get, id=project_id)
            if (getattr(project, "model_version", "") or "") == want:
                continue
            await asyncio.to_thread(
                ls.projects.update, id=project_id, model_version=want
            )
            updated += 1
            activity.logger.info(
                "project %s: model_version -> %s (predictions now visible)",
                project_id,
                want,
            )
        except Exception as exc:  # pylint: disable=broad-except
            # Never fail the backfill over display configuration: the
            # predictions are attached and correct either way. Logged with the
            # exception rather than just its type -- a persistently failing
            # update (missing write scope, 404, a rejected value) leaves the
            # bug this fixes live, and the type alone cannot tell you which.
            # Same reasoning as `heal_labeling_config`.
            activity.logger.warning(
                "project %s: could not set model_version to %r: %s",
                project_id,
                want,
                exc,
            )
    return updated


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
        # Which tiers each project actually holds, counted over every placeable
        # prediction rather than only the newly attached ones -- on a re-run
        # nothing is attached and the project would otherwise be left pointing
        # nowhere.
        tags_by_project: Dict[int, Counter] = {}
        for image_id, (task_id, project_id) in targets.items():
            wrapper = prediction_annotations(prediction_by_image[image_id])
            if not wrapper:
                continue
            body = wrapper[0]
            tags_by_project.setdefault(project_id, Counter())[
                body["model_version"]
            ] += 1
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

        await _ensure_projects_show_predictions(ls, tags_by_project)

    activity.logger.info(
        "dive %d: attached %d head/tail prediction(s) to existing tasks",
        dive_id,
        attached,
    )
    return attached
