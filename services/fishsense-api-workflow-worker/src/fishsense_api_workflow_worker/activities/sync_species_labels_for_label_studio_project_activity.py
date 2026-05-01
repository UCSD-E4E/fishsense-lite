"""Activity to sync species labels for a Label Studio project (stage 4.2).

Mirrors the laser/headtail/dive-slate sync pattern. The species
annotation is the richest of the four — beyond the laser keypoint it
also carries grouping, slate-upside-down, and four taxonomy fields
(content, fish_measurable, fish_angle, fish_curved). All are pulled
into their dedicated SpeciesLabel columns so downstream stages
(especially 6.1's `update_dive_image_groups`) can read them without
re-parsing `label_studio_json`.

Notebook this ports: `scripts/stage4.2_sync_species_labels.ipynb`. The
notebook resolved annotators by parsing the email out of
`created_username`; here we use `task.annotators[-1]` -> SDK
`get_by_label_studio_id` to match the other three sync activities and
ride the existing user-sync upstream of every sync run.
"""

import json
from typing import Any, Dict

from fishsense_api_sdk.client import Client
from fishsense_api_sdk.models.species_label import SpeciesLabel
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import (
    SYNC_CONCURRENCY,
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


def _laser_keypoint(results: list[dict]) -> Dict[str, Any] | None:
    """Pull the laser keypoint section (with its original_width/height) or None."""
    for r in results:
        if r["from_name"] == "laser":
            return r
    return None


def _parse_results(annotation: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the species annotation fields out of an LS task result list.

    Returns absolute pixel coordinates for laser_x/y (the LS
    annotation reports percentages of original_width/height — multiply
    out so downstream consumers see the same units the laser-only
    sync writes).
    """
    results = annotation.get("result") or []

    grouping = _first_choice(results, "grouping")

    exclude_choice = _first_choice(results, "exclude")
    top_three_photos_of_group = (
        exclude_choice == "Top 3 photos of group"
        if exclude_choice is not None
        else None
    )

    slate_choice = _first_choice(results, "slate")
    slate_upside_down = (
        slate_choice == "Slate upside down" if slate_choice is not None else None
    )

    laser_x: float | None = None
    laser_y: float | None = None
    laser_label: str | None = None
    laser_section = _laser_keypoint(results)
    if laser_section is not None:
        original_width = laser_section.get("original_width")
        original_height = laser_section.get("original_height")
        value = laser_section.get("value", {})
        if (
            original_width is not None
            and original_height is not None
            and "x" in value
            and "y" in value
        ):
            laser_x = value["x"] * original_width / 100.0
            laser_y = value["y"] * original_height / 100.0
        keypointlabels = value.get("keypointlabels") or []
        if keypointlabels:
            laser_label = keypointlabels[0]

    return {
        "grouping": grouping,
        "top_three_photos_of_group": top_three_photos_of_group,
        "slate_upside_down": slate_upside_down,
        "laser_x": laser_x,
        "laser_y": laser_y,
        "laser_label": laser_label,
        "content_of_image": _content_of_image(results),
        "fish_measurable_category": _first_taxonomy_leaf(results, "measurable"),
        "fish_angle_category": _first_taxonomy_leaf(results, "fishAngles"),
        "fish_curved_category": _first_taxonomy_leaf(results, "fishCurve"),
    }


def _apply_parsed(species_label: SpeciesLabel, parsed: Dict[str, Any]) -> None:
    """Copy parser output into the SpeciesLabel, leaving stored values
    in place when the annotation didn't specify a field.

    Mirror's the notebook's "if section is present, write it" semantics:
    a re-sync of an annotation that dropped (say) the laser keypoint
    won't clobber the previously-recorded laser_x/y. Choices fields
    follow the same rule via the `is not None` gate.
    """
    if parsed["grouping"] is not None:
        species_label.grouping = parsed["grouping"]
    if parsed["top_three_photos_of_group"] is not None:
        species_label.top_three_photos_of_group = parsed["top_three_photos_of_group"]
    if parsed["slate_upside_down"] is not None:
        species_label.slate_upside_down = parsed["slate_upside_down"]
    if parsed["laser_x"] is not None:
        species_label.laser_x = parsed["laser_x"]
    if parsed["laser_y"] is not None:
        species_label.laser_y = parsed["laser_y"]
    if parsed["laser_label"] is not None:
        species_label.laser_label = parsed["laser_label"]
    if parsed["content_of_image"] is not None:
        species_label.content_of_image = parsed["content_of_image"]
    if parsed["fish_measurable_category"] is not None:
        species_label.fish_measurable_category = parsed["fish_measurable_category"]
    if parsed["fish_angle_category"] is not None:
        species_label.fish_angle_category = parsed["fish_angle_category"]
    if parsed["fish_curved_category"] is not None:
        species_label.fish_curved_category = parsed["fish_curved_category"]


async def _update_species_label(fs: Client, task: Any) -> None:
    species_label = await fs.labels.get_species_label(label_studio_id=task.id)
    if species_label is None:
        return

    if task.annotators:
        user = await fs.users.get_by_label_studio_id(task.annotators[-1])
        species_label.user_id = user.id

    species_label.label_studio_json = json.loads(task.json())
    species_label.completed = task.is_labeled
    species_label.updated_at = task.updated_at

    if task.annotations:
        parsed = _parse_results(task.annotations[0])
        _apply_parsed(species_label, parsed)

    await fs.labels.put_species_label(species_label.image_id, species_label)


@activity.defn
async def sync_species_labels_for_label_studio_project_activity(project_id: int):
    """Activity to sync species labels for a Label Studio project."""
    await sync_label_studio_project(
        project_id, _update_species_label, kind="species"
    )
