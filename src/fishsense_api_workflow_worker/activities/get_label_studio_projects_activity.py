"""Activity to get Label Studio projects."""

import asyncio
from typing import List, Tuple

from label_studio_sdk.client import LabelStudio
from temporalio import activity

from fishsense_api_workflow_worker.activities.label_utils import (
    get_dive_slate_labels,
    get_headtail_labels,
    get_laser_labels,
    get_species_labels,
)
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
        name=project.description,
    )


@activity.defn
async def get_label_studio_projects_activity() -> Tuple[
    List[LabelStudioProject],
    List[LabelStudioProject],
    List[LabelStudioProject],
    List[LabelStudioProject],
]:
    """Activity to get Label Studio projects for laser, species, head-tail, and slate labels."""
    ls = get_ls_client()

    async with get_fs_client() as fs:
        laser_labels = await get_laser_labels(fs)
        species_labels = await get_species_labels(fs)
        head_tail_labels = await get_headtail_labels(fs)
        slate_labels = await get_dive_slate_labels(fs)

        laser_label_project_project_id = {
            label.label_studio_project_id
            for label in laser_labels
            if not label.completed
        }

        laser_label_studio_projects = await asyncio.gather(
            *(
                __get_label_studio_project(ls, project_id)
                for project_id in laser_label_project_project_id
            )
        )

        species_label_project_project_id = {
            label.label_studio_project_id
            for label in species_labels
            if not label.completed
        }

        species_label_studio_projects = await asyncio.gather(
            *(
                __get_label_studio_project(ls, project_id)
                for project_id in species_label_project_project_id
            )
        )

        head_tail_labels_project_project_id = {
            label.label_studio_project_id
            for label in head_tail_labels
            if not label.completed
        }

        head_tail_label_studio_projects = await asyncio.gather(
            *(
                __get_label_studio_project(ls, project_id)
                for project_id in head_tail_labels_project_project_id
            )
        )

        slate_labels_project_project_id = {
            label.label_studio_project_id
            for label in slate_labels
            if not label.completed
        }

        slate_label_studio_projects = await asyncio.gather(
            *(
                __get_label_studio_project(ls, project_id)
                for project_id in slate_labels_project_project_id
            )
        )

        return (
            laser_label_studio_projects,
            species_label_studio_projects,
            head_tail_label_studio_projects,
            slate_label_studio_projects,
        )
