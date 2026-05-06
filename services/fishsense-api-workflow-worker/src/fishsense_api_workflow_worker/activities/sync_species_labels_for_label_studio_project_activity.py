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
