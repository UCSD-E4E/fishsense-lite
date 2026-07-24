"""Activity to sync species labels for a Label Studio project (stage 4.2).

Mirrors the laser/headtail/dive-slate sync pattern. As of 2026-05-05
the species LS XML no longer carries the laser keypoint or the
"Slate upside down" choice, so this activity reads only the still-
present fields: `grouping`, `exclude` (top-3 marker), the species
taxonomy, and the three fish-attribute taxonomies (measurable, angle,
curve). The unused SpeciesLabel columns (`laser_x`, `laser_y`,
`laser_label`, `slate_upside_down`) are left whatever they were —
historical rows keep their values; new rows from the new XML never
populate them.

Notebook this ports: `scripts/stage4.2_sync_species_labels.ipynb`. The
notebook resolved annotators by parsing the email out of
`created_username`; here we use `task.annotators[-1]` -> SDK
`get_by_label_studio_id` to match the other three sync activities and
ride the existing user-sync upstream of every sync run.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict

from fishsense_api_sdk.client import Client
from fishsense_api_sdk.models.species_label import SpeciesLabel
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import (
    SYNC_CONCURRENCY,
    _coerce_updated_at,
    get_fs_client,
    resolve_annotator_user,
    sync_label_studio_project,
)

__all__ = [
    "sync_species_labels_for_label_studio_project_activity",
    "SYNC_CONCURRENCY",
]


def _first_choice(results: list[dict], from_name: str) -> str | None:
    """First `value.choices[0]` from results matching `from_name`, or None."""
    for r in results:
        if r["from_name"] == from_name:
            choices = r.get("value", {}).get("choices") or []
            if choices:
                return choices[0]
    return None


def _first_taxonomy_leaf(results: list[dict], from_name: str) -> str | None:
    """First `value.taxonomy[0][0]` from results matching `from_name`.

    The notebook uses `taxonomy[0][0]` for fish_measurable / fish_angle /
    fish_curved — the leaf of the first taxonomy path. Keep that shape.
    """
    for r in results:
        if r["from_name"] == from_name:
            taxonomy = r.get("value", {}).get("taxonomy") or []
            if taxonomy and taxonomy[0]:
                return taxonomy[0][0]
    return None


def _content_of_image(results: list[dict]) -> str | None:
    """Join the species taxonomy path with ", ".

    Notebook: `", ".join(taxonomy[0])`. Stage 14's
    `_parse_species_names` reads this back to derive (common,
    scientific) from the trailing element — keep the join character
    fixed.
    """
    for r in results:
        if r["from_name"] == "species":
            taxonomy = r.get("value", {}).get("taxonomy") or []
            if taxonomy:
                return ", ".join(taxonomy[0])
    return None


def _parse_results(annotation: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the species annotation fields out of an LS task result list."""
    results = annotation.get("result") or []

    grouping = _first_choice(results, "grouping")

    exclude_choice = _first_choice(results, "exclude")
    top_three_photos_of_group = (
        exclude_choice == "Top 3 photos of group"
        if exclude_choice is not None
        else None
    )

    return {
        "grouping": grouping,
        "top_three_photos_of_group": top_three_photos_of_group,
        "content_of_image": _content_of_image(results),
        "fish_measurable_category": _first_taxonomy_leaf(results, "measurable"),
        "fish_angle_category": _first_taxonomy_leaf(results, "fishAngles"),
        "fish_curved_category": _first_taxonomy_leaf(results, "fishCurve"),
    }


def _apply_parsed(species_label: SpeciesLabel, parsed: Dict[str, Any]) -> None:
    """Copy parser output into the SpeciesLabel, leaving stored values
    in place when the annotation didn't specify a field."""
    if parsed["grouping"] is not None:
        species_label.grouping = parsed["grouping"]
    if parsed["top_three_photos_of_group"] is not None:
        species_label.top_three_photos_of_group = parsed["top_three_photos_of_group"]
    if parsed["content_of_image"] is not None:
        species_label.content_of_image = parsed["content_of_image"]
    if parsed["fish_measurable_category"] is not None:
        species_label.fish_measurable_category = parsed["fish_measurable_category"]
    if parsed["fish_angle_category"] is not None:
        species_label.fish_angle_category = parsed["fish_angle_category"]
    if parsed["fish_curved_category"] is not None:
        species_label.fish_curved_category = parsed["fish_curved_category"]


def _slate_type_choice(
    results: list[dict], valid_slate_names: set[str]
) -> str | None:
    """The DiveSlate-template name a labeler picked, or None.

    The species Taxonomy carries the slate type (H-Slate / V-Slate N /
    Tic-Tac-Toe N) as its own leaf under `Slate`, alongside `Laser on
    slate`. `content_of_image` only keeps `taxonomy[0]` (the laser
    marker, which stage 14 parses), so the slate-type path is dropped
    there. Here we scan *all* taxonomy paths and return the leaf that
    matches a real DiveSlate template name — that's the "which slate"
    answer we map to `dive_slate_id`.
    """
    for r in results:
        if r.get("from_name") == "species":
            for path in r.get("value", {}).get("taxonomy") or []:
                if path and path[-1] in valid_slate_names:
                    return path[-1]
    return None


def _reduce_slate_winners(
    votes: list[tuple[int, datetime | None, int]],
) -> dict[int, int]:
    """Collapse per-image slate votes to one `dive_slate_id` per dive.

    `votes` is `(dive_id, updated_at, dive_slate_id)`. Most-recent
    completed annotation wins (a re-label with a newer timestamp
    overrides an older one); a vote with no timestamp never displaces
    one that has a timestamp.
    """
    best: dict[int, tuple[datetime | None, int]] = {}
    for dive_id, ts, slate_id in votes:
        current = best.get(dive_id)
        if current is None or (
            ts is not None and (current[0] is None or ts > current[0])
        ):
            best[dive_id] = (ts, slate_id)
    return {dive_id: slate_id for dive_id, (_, slate_id) in best.items()}


async def _update_species_label(fs: Client, task: Any) -> int | None:
    species_label = await fs.labels.get_species_label(label_studio_id=task.id)
    if species_label is None:
        return None

    # Attribution is best-effort: hosted LS returns `annotators` as
    # dicts rather than ints, and mis-handling that used to 422 and
    # kill the whole project's sync. See resolve_annotator_user.
    user = await resolve_annotator_user(fs, task)
    if user is not None:
        species_label.user_id = user.id

    species_label.label_studio_json = json.loads(task.json())
    species_label.completed = task.is_labeled
    species_label.updated_at = task.updated_at

    if task.annotations:
        parsed = _parse_results(task.annotations[0])
        _apply_parsed(species_label, parsed)

    await fs.labels.put_species_label(species_label.image_id, species_label)
    return species_label.image_id


async def _resolve_dive_slates(
    completed: list[tuple[int, datetime | None, list[dict]]],
) -> None:
    """Set `dive_slate_id` from the slate type labelers picked.

    `completed` is `(image_id, updated_at, annotation_results)` for every
    completed task this sync processed. Maps each slate-type choice to a
    DiveSlate id, resolves the image's dive, and writes the most-recent
    winner per dive. No-op when nothing was completed or no slate type was
    chosen — so it never touches a dive whose labelers didn't identify a
    slate this run.
    """
    if not completed:
        return

    async with get_fs_client() as fs:
        slates = await fs.dive_slates.get() or []
        name_to_id = {slate.name: slate.id for slate in slates}
        valid_names = set(name_to_id)

        votes: list[tuple[int, datetime | None, int]] = []
        dive_by_image: dict[int, int | None] = {}
        for image_id, ts, results in completed:
            slate_name = _slate_type_choice(results, valid_names)
            if slate_name is None:
                continue
            if image_id not in dive_by_image:
                image = await fs.images.get(image_id=image_id)
                dive_by_image[image_id] = image.dive_id if image else None
            dive_id = dive_by_image[image_id]
            if dive_id is not None:
                votes.append((dive_id, ts, name_to_id[slate_name]))

        for dive_id, slate_id in _reduce_slate_winners(votes).items():
            await fs.dives.set_dive_slate(dive_id, slate_id)
            activity.logger.info(
                "species sync set dive_slate dive_id=%d dive_slate_id=%d",
                dive_id,
                slate_id,
            )


@activity.defn
async def sync_species_labels_for_label_studio_project_activity(project_id: int):
    """Activity to sync species labels for a Label Studio project.

    Besides writing each SpeciesLabel, this collects the slate-type
    choice from every *completed* task and, after the sync, sets each
    dive's `dive_slate_id` (most-recent-completed wins). That's the only
    thing in the pipeline that populates `dive_slate_id` from labeler
    input — stages 9/12/13 read it but nothing else writes it.
    """
    completed: list[tuple[int, datetime | None, list[dict]]] = []
    lock = asyncio.Lock()

    async def _update(fs: Client, task: Any) -> None:
        image_id = await _update_species_label(fs, task)
        if image_id is None or not getattr(task, "is_labeled", False):
            return
        annotations = task.annotations or []
        if not annotations:
            return
        results = annotations[0].get("result") or []
        ts = _coerce_updated_at(getattr(task, "updated_at", None))
        async with lock:
            completed.append((image_id, ts, results))

    await sync_label_studio_project(project_id, _update, kind="species")

    await _resolve_dive_slates(completed)
