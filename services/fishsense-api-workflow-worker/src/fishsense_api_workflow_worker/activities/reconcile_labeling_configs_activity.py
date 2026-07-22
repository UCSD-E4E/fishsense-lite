"""Push the current `<STAGE>_LABELING_CONFIG_XML` onto every per-dive project.

`create_or_get_label_studio_project` already heals a project's labeling
config when it finds one, but it only runs during **populate** — and
populate stops dispatching for a dive once every image has a species row and
the dive leaves the cohort. So the heal reaches projects that are still
filling and never reaches finished ones, which are exactly the stable
projects labelers spend their time in.

Observed 2026-07-21 after the Fish Model taxonomy swap: all 11 species
projects whose dive was still in `needing-species-population` picked up the
new choices, while `082923_FishModels_FSL02 #58 - Species Labeling`
(pid 274353, fully populated, out of cohort) stayed on the old list
indefinitely. Every future taxonomy edit would strand it again.

This activity closes that gap by walking the projects directly rather than
riding a cohort: it enumerates the workspace, matches each project's title
suffix to the stage that owns it, and delegates to the same
`heal_labeling_config` used by the populate path — so drift detection,
the no-op-when-unchanged behaviour, and the swallow-on-LS-rejection
semantics are all shared rather than reimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass

from temporalio import activity

from fishsense_api_workflow_worker.activities.create_dive_slate_label_studio_project_activity import (  # pylint: disable=line-too-long
    DIVE_SLATE_LABELING_CONFIG_XML,
    DIVE_SLATE_PROJECT_TITLE_SUFFIX,
)
from fishsense_api_workflow_worker.activities.create_headtail_label_studio_project_activity import (  # pylint: disable=line-too-long
    HEADTAIL_LABELING_CONFIG_XML,
    HEADTAIL_PROJECT_TITLE_SUFFIX,
)
from fishsense_api_workflow_worker.activities.create_laser_label_studio_project_activity import (  # pylint: disable=line-too-long
    LASER_LABELING_CONFIG_XML,
    LASER_PROJECT_TITLE_SUFFIX,
)
from fishsense_api_workflow_worker.activities.create_species_label_studio_project_activity import (  # pylint: disable=line-too-long
    SPECIES_LABELING_CONFIG_XML,
    SPECIES_PROJECT_TITLE_SUFFIX,
)
from fishsense_api_workflow_worker.activities.populate_utils import (
    _get_ls_client,
    _resolve_workspace_id,
    heal_labeling_config,
)

# Title suffix -> the labeling config that owns it. Per-dive titles are
# `"{dive.name} #{dive_id} - {suffix}"`, so the suffix is what identifies the
# stage. Ordered longest-first so a suffix that is a substring of another
# can't shadow it.
_CONFIG_BY_SUFFIX = {
    LASER_PROJECT_TITLE_SUFFIX: LASER_LABELING_CONFIG_XML,
    SPECIES_PROJECT_TITLE_SUFFIX: SPECIES_LABELING_CONFIG_XML,
    HEADTAIL_PROJECT_TITLE_SUFFIX: HEADTAIL_LABELING_CONFIG_XML,
    DIVE_SLATE_PROJECT_TITLE_SUFFIX: DIVE_SLATE_LABELING_CONFIG_XML,
}


@dataclass
class ReconcileLabelingConfigsResult:
    """Counts for one reconcile pass."""

    scanned: int = 0
    healed: int = 0
    unchanged: int = 0
    unrecognized: int = 0


def _config_for_title(title: str | None) -> str | None:
    """The labeling config owning `title`, or None if it isn't a per-dive one.

    Matched on the suffix rather than parsed, so demo projects and anything
    else sharing the workspace are left alone.
    """
    if not title:
        return None
    for suffix, config in sorted(
        _CONFIG_BY_SUFFIX.items(), key=lambda kv: -len(kv[0])
    ):
        if title.endswith(suffix):
            return config
    return None


@activity.defn
async def reconcile_labeling_configs_activity() -> ReconcileLabelingConfigsResult:
    """Converge every per-dive project onto its stage's current config."""
    ls = _get_ls_client()
    workspace_id = _resolve_workspace_id(ls)

    projects = list(
        ls.projects.list(workspaces=[workspace_id])
        if workspace_id is not None
        else ls.projects.list()
    )

    result = ReconcileLabelingConfigsResult()
    for project in projects:
        title = getattr(project, "title", None)
        config = _config_for_title(title)
        if config is None:
            result.unrecognized += 1
            continue

        result.scanned += 1
        activity.heartbeat()
        if await heal_labeling_config(ls, project, config):
            result.healed += 1
            activity.logger.info(
                "reconciled labeling config for project id=%s title=%r",
                getattr(project, "id", None),
                title,
            )
        else:
            result.unchanged += 1

    activity.logger.info(
        "labeling-config reconcile: scanned=%d healed=%d unchanged=%d "
        "unrecognized=%d",
        result.scanned,
        result.healed,
        result.unchanged,
        result.unrecognized,
    )
    return result
