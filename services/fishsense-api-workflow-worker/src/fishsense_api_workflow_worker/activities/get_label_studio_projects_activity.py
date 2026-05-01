"""Activity to get Label Studio projects."""

import asyncio
from typing import List, Tuple

from label_studio_sdk.client import LabelStudio
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client, get_ls_client
from fishsense_api_workflow_worker.models.label_studio_project import LabelStudioProject


async def __get_label_studio_project(
    ls: LabelStudio, label_studio_project_id: int
) -> LabelStudioProject:
    """Get Label Studio project by ID.

    Args:
        ls (LabelStudio): Label Studio client
        label_studio_project_id (int): Label Studio project ID

    Returns:
        LabelStudioProject: Label Studio project
    """
    project = await asyncio.to_thread(ls.projects.get, label_studio_project_id)
    return LabelStudioProject(
        id=project.id,
        name=project.title,
    )


@activity.defn
async def get_label_studio_projects_activity() -> Tuple[
    List[LabelStudioProject],
    List[LabelStudioProject],
    List[LabelStudioProject],
    List[LabelStudioProject],
]:
    """Activity to get Label Studio projects for laser, species, head-tail, and slate labels.

    Pulls only the distinct *incomplete* project IDs from the api per
    label kind (4 small HTTP calls — server-side `SELECT DISTINCT` with
    a `completed IS NULL OR completed = false` filter), then resolves
    each project's display name out of Label Studio. Replaces the old
    per-dive fan-out that paged every label across the wire.
    """
    ls = get_ls_client()

    async with get_fs_client() as fs:
        activity.logger.info(
            "Getting Label Studio projects for laser, species, head-tail, and slate labels"
        )
        (
            laser_project_ids,
            species_project_ids,
            head_tail_project_ids,
            slate_project_ids,
        ) = await asyncio.gather(
            fs.labels.get_laser_label_studio_project_ids(incomplete=True),
            fs.labels.get_species_label_studio_project_ids(incomplete=True),
            fs.labels.get_headtail_label_studio_project_ids(incomplete=True),
            fs.labels.get_dive_slate_label_studio_project_ids(incomplete=True),
        )

    activity.logger.info(f"Laser label project IDs: {laser_project_ids}")
    activity.logger.info(f"Species label project IDs: {species_project_ids}")
    activity.logger.info(f"Head-tail label project IDs: {head_tail_project_ids}")
    activity.logger.info(f"Slate label project IDs: {slate_project_ids}")

    (
        laser_label_studio_projects,
        species_label_studio_projects,
        head_tail_label_studio_projects,
        slate_label_studio_projects,
    ) = await asyncio.gather(
        asyncio.gather(
            *(__get_label_studio_project(ls, pid) for pid in laser_project_ids)
        ),
        asyncio.gather(
            *(__get_label_studio_project(ls, pid) for pid in species_project_ids)
        ),
        asyncio.gather(
            *(__get_label_studio_project(ls, pid) for pid in head_tail_project_ids)
        ),
        asyncio.gather(
            *(__get_label_studio_project(ls, pid) for pid in slate_project_ids)
        ),
    )

    return (
        list(laser_label_studio_projects),
        list(species_label_studio_projects),
        list(head_tail_label_studio_projects),
        list(slate_label_studio_projects),
    )
