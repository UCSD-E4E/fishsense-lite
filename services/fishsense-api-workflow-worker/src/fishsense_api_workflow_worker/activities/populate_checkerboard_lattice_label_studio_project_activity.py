"""Import rendered lattices into the verification project as LS tasks.

Turns the data-worker's `CheckerboardLatticeRender` list into tasks carrying
the overlay JPEG and, inline, the detected corners as keypoint predictions.

**Predictions must be inline at import, never backfilled.** Label Studio
surfaces predictions only for the version named in the *project's*
`model_version`, and `import_tasks` sets that field for free when the tasks it
imports carry `predictions` — which is exactly why the laser and head/tail
backfill paths need `ensure_project_shows_predictions` and this one does not.
That helper would not help here anyway: it only touches per-dive projects,
identified by the `#{dive_id}` marker in the title, and this study runs one
shared project with no dive in its name. So the invariant is structural rather
than enforced elsewhere — attach predictions at import or the labeler sees a
blank frame, and would read that as "nothing was detected".

**No label row is written.** Every other populate in this repo anchors an
(image, task, project) triple in a `*Label` table, and the import helper takes
a `record_label` hook to do it. There is no `CheckerboardLatticeLabel` model
and deliberately so: this is a diagnostic that answers one question once, and
a schema plus a migration plus an SDK mirror to hold a few hundred throwaway
verdicts would be permanent weight for temporary work. The verdicts are read
back off the Label Studio annotations. The hook is therefore a no-op, but the
helper is still used, because what it really carries is the dedupe-by-URL and
the hosted-LS async-import handling that this repo paid for in duplicated
tasks (ten projects at two tasks per image, one at twenty-three).
"""

from __future__ import annotations

from typing import Any, Iterable, List

from fishsense_shared import CheckerboardLatticeRender
from fishsense_shared.object_store import CHECKERBOARD_LATTICE_JPEG_FOLDER
from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    build_image_url,
    import_tasks_and_record_labels,
    publish_label_studio_project,
)

__all__ = [
    "LATTICE_FOLDER",
    "LATTICE_MODEL_VERSION",
    "lattice_predictions",
    "populate_checkerboard_lattice_label_studio_project_activity",
    "renders_worth_labeling",
]

LATTICE_FOLDER = CHECKERBOARD_LATTICE_JPEG_FOLDER

# Must match the labeling config's control names — see
# `create_checkerboard_lattice_label_studio_project_activity`. A mismatch is
# silent: the prediction is stored and nothing renders.
_KEYPOINT_FROM_NAME = "lattice"
_KEYPOINT_TO_NAME = "image"
_KEYPOINT_LABEL = "Detected corner"

#: Names the tier these predictions belong to. One fixed value rather than the
#: laser path's versioned tags: there is only ever one detector behind this
#: study, and a moving tag would split one project's predictions across
#: versions for no gain.
LATTICE_MODEL_VERSION = "checkerboard-lattice-v1"


def _is_placeable(render: CheckerboardLatticeRender) -> bool:
    return bool(render.corners) and bool(render.width) and bool(render.height)


def renders_worth_labeling(
    renders: Iterable[CheckerboardLatticeRender],
) -> List[CheckerboardLatticeRender]:
    """The renders that should become tasks.

    Drops frames the detector rejected, and frames that came back rendered but
    with nothing placeable. Both would reach a labeler as an image with no
    marks on it, whose only honest verdict is "there is nothing here to judge"
    — an answer about the detector's hit rate, which this study is not asking,
    and which would be indistinguishable afterwards from a real lattice fault.
    """
    return [r for r in renders if r.skip_reason is None and _is_placeable(r)]


def lattice_predictions(render: CheckerboardLatticeRender) -> list:
    """The LS `predictions` list for one render, or [] when unplaceable.

    One keypoint result per detected corner. They are redundant with the
    corners already burned into the JPEG, and that is intended: the drawn
    edges are what make a coarse lattice legible at a glance, while the
    predictions are what let a labeler toggle the marks off and check the bare
    board underneath.
    """
    if not _is_placeable(render):
        return []

    width = float(render.width)
    height = float(render.height)
    return [
        {
            "model_version": LATTICE_MODEL_VERSION,
            "result": [
                {
                    "from_name": _KEYPOINT_FROM_NAME,
                    "to_name": _KEYPOINT_TO_NAME,
                    "type": "keypointlabels",
                    "original_width": render.width,
                    "original_height": render.height,
                    "image_rotation": 0,
                    "value": {
                        # Percentages, converted from the rectified pixels the
                        # detector reported. Pixels here scatter the marks with
                        # no error, and a labeler would report the units bug as
                        # a lattice fault.
                        "x": float(x) / width * 100,
                        "y": float(y) / height * 100,
                        "width": 0.3,
                        "keypointlabels": [_KEYPOINT_LABEL],
                    },
                }
                for x, y in render.corners
            ],
        }
    ]


def _build_task(render: CheckerboardLatticeRender) -> dict:
    """One LS task: the overlay JPEG plus its corners as a prediction.

    **The data payload carries the image and nothing else, and that is a
    blinding requirement rather than minimalism.** Every key here becomes a
    sortable, filterable column in the Label Studio Data Manager, so anything
    describing the detection is handed to the person being asked to judge it:

    * `median_spacing_px` *is* the diagnostic. A coarse lattice has roughly
      twice the spacing of a good one, so one sort groups the faults before
      anybody has looked at a frame.
    * `detected_rows`/`cols` leak the same thing more weakly — a lattice at
      double pitch reports about half the grid.
    * `image_id` is monotonic within a dive, so sorting on it reassembles the
      per-dive blocks the shuffle exists to break up.

    `build_task_data` is not reused for the same reason: it emits `taken` and
    `image_id` precisely so a project *can* be sorted back into capture order.

    Traceability survives anyway. The checksum is in the image URL, which is
    how every populate path here identifies a frame, and checksums sort
    randomly — so a verdict maps back to its dive afterwards without the task
    ever exposing an ordering a labeler could exploit.
    """
    url = build_image_url(LATTICE_FOLDER, render.checksum)
    return {
        "data": {"image": url, "img": url},
        "predictions": lattice_predictions(render),
    }


async def _record_nothing(_item: Any, _task_id: int) -> None:
    """No label row for this stage — see the module docstring."""
    return None


@activity.defn
async def populate_checkerboard_lattice_label_studio_project_activity(
    project_id: int,
    renders: List[CheckerboardLatticeRender],
) -> int:
    """Import the rendered lattices as tasks. Returns the count imported.

    `project_id` is passed in rather than resolved here. Every other stage has
    its workflow call the Create activity and hand the id down, and the
    separation earns its keep: creating the project and importing hundreds of
    tasks are different failure modes, and folding them into one activity puts
    project creation inside the import's retry boundary.
    """
    renders = [
        (
            r
            if isinstance(r, CheckerboardLatticeRender)
            else CheckerboardLatticeRender.model_validate(r)
        )
        for r in renders
    ]
    worth = renders_worth_labeling(renders)
    activity.logger.info(
        "lattice verification: %d of %d renders worth labeling",
        len(worth),
        len(renders),
    )
    if not worth:
        return 0

    result = await import_tasks_and_record_labels(
        project_id=project_id,
        tasks=[_build_task(r) for r in worth],
        record_label=_record_nothing,
        items=worth,
    )
    # `.recorded`, not the result itself: `import_tasks_and_record_labels`
    # returns an `ImportResult(recorded, deferred)` NamedTuple, and returning it
    # from an activity annotated `-> int` serialises as the two-element list
    # `[recorded, deferred]` with no error anywhere.
    #
    # Publishing is gated on `.complete`, as every other caller gates it.
    # Hosted LS imports asynchronously, so a deferred batch means tasks exist
    # that are not yet listable; publishing then hands annotators a
    # half-populated project. The scheduled stages tolerate that because their
    # next firing reconciles it — this study is on-demand and has no next
    # firing, so an unnoticed partial publish is what a labeler would work
    # through.
    if result.complete:
        await publish_label_studio_project(project_id)
    else:
        activity.logger.error(
            "lattice verification: %d task(s) deferred by LS; leaving "
            "project_id=%d unpublished. Re-run to reconcile.",
            result.deferred,
            project_id,
        )
    activity.logger.info(
        "lattice verification: imported %s tasks into project_id=%d",
        result.recorded,
        project_id,
    )
    return result.recorded
