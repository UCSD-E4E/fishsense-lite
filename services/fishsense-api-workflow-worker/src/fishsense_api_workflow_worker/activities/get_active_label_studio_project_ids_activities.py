"""Query activities — return the set of LS projects to populate.

Each populate workflow needs to know which Label Studio project(s) to
push tasks to. Rather than carry a static config ID, we query SQL for
the projects that already hold incomplete labels of the right kind —
i.e. projects that are actively being labeled. New projects without
any labels yet are out of scope here; the separate
project-creation workflow is responsible for seeding them.

The four activities are thin wrappers over the SDK methods:
  * `LabelClient.get_laser_label_studio_project_ids(incomplete=True)`
  * `LabelClient.get_species_label_studio_project_ids(incomplete=True)`
  * `LabelClient.get_headtail_label_studio_project_ids(incomplete=True)`
  * `LabelClient.get_dive_slate_label_studio_project_ids(incomplete=True)`

`incomplete=True` filters out fully-completed legacy projects (e.g.
historical archives whose labels are all done) so populate doesn't
spam tasks into projects that are intentionally retired.
"""

from typing import List

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def get_active_laser_label_studio_project_ids_activity() -> List[int]:
    """Return LS project IDs with at least one incomplete laser label."""
    async with get_fs_client() as fs:
        return await fs.labels.get_laser_label_studio_project_ids(incomplete=True)


@activity.defn
async def get_active_species_label_studio_project_ids_activity() -> List[int]:
    """Return LS project IDs with at least one incomplete species label."""
    async with get_fs_client() as fs:
        return await fs.labels.get_species_label_studio_project_ids(incomplete=True)


@activity.defn
async def get_active_headtail_label_studio_project_ids_activity() -> List[int]:
    """Return LS project IDs with at least one incomplete headtail label."""
    async with get_fs_client() as fs:
        return await fs.labels.get_headtail_label_studio_project_ids(
            incomplete=True
        )


@activity.defn
async def get_active_dive_slate_label_studio_project_ids_activity() -> List[int]:
    """Return LS project IDs with at least one incomplete dive-slate label."""
    async with get_fs_client() as fs:
        return await fs.labels.get_dive_slate_label_studio_project_ids(
            incomplete=True
        )
