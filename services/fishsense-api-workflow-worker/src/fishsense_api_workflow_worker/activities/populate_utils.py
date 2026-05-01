"""Shared helpers for populate / create LS-project activities.

Each populate stage creates LS tasks pointing at the labeler-facing
static-file server, imports them in one batch, then upserts a
per-image label row anchoring the (image, LS task, LS project) triple.
Each create stage uses `create_or_get_label_studio_project` to
idempotently materialize the LS project from a stored labeling-config
XML — populate workflows themselves query SQL for the project IDs to
target and never create projects.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Iterable, List

from label_studio_sdk.client import LabelStudio
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_ls_client
from fishsense_api_workflow_worker.config import settings


def build_image_url(folder: str, checksum: str) -> str:
    """Build the labeler-facing URL for a preprocessed JPEG.

    Folder names match the GET routes already exposed by
    `deploy/static_file_server/nginx.conf` (preprocess_jpeg,
    groups_jpeg, headtail_jpeg, dive_slate_jpgs).
    """
    base = settings.static_files.public_url_base.rstrip("/")
    return f"{base}/api/v1/data/{folder}/{checksum}"


async def create_or_get_label_studio_project(
    *,
    project_title: str,
    labeling_config_xml: str,
) -> int:
    """Idempotent create — return the LS project ID for `project_title`,
    creating one with `labeling_config_xml` if none exists.

    Used by the create-side workflows. The populate-side workflows
    don't call this; they query SQL for actively-labeled project IDs
    via the `get_active_*_label_studio_project_ids_activity` family.
    """
    ls = _get_ls_client()

    matches = await asyncio.to_thread(
        lambda: [p for p in ls.projects.list() if p.title == project_title]
    )
    if matches:
        if len(matches) > 1:
            activity.logger.warning(
                "Multiple LS projects titled %r; using id=%d",
                project_title,
                matches[0].id,
            )
        return matches[0].id

    if not labeling_config_xml:
        raise RuntimeError(
            f"Cannot create LS project {project_title!r}: the labeling-"
            "config XML constant is empty. Paste the labeling-config XML "
            "from your existing prod project (Project Settings -> Labeling "
            "Interface -> Code) into the corresponding constant."
        )

    project = await asyncio.to_thread(
        ls.projects.create,
        title=project_title,
        label_config=labeling_config_xml,
    )
    activity.logger.info(
        "Created LS project %r (id=%d)", project_title, project.id
    )
    return project.id


async def import_tasks_and_record_labels(
    *,
    project_id: int,
    tasks: List[dict],
    record_label: Callable[[Any, int], Awaitable[None]],
    items: Iterable[Any],
) -> int:
    """Import `tasks` to LS, then PUT one label row per (item, task_id).

    `record_label(item, task_id)` is the per-stage hook that builds and
    upserts a LaserLabel/SpeciesLabel/HeadTailLabel/DiveSlateLabel row
    via the SDK. Rows are PUT in parallel via TaskGroup matching the
    notebook behaviour; the api PUTs are upserts so partial-failure
    replay is safe.
    """
    if not tasks:
        return 0

    ls = _get_ls_client()
    imported = await asyncio.to_thread(
        lambda: ls.projects.import_tasks(
            project_id, request=tasks, return_task_ids=True
        )
    )
    task_ids: List[int] = list(imported.task_ids)

    items_list = list(items)
    if len(task_ids) != len(items_list):
        raise RuntimeError(
            f"LS import_tasks returned {len(task_ids)} task IDs for "
            f"{len(items_list)} input items; refusing to write mismatched "
            "label rows"
        )

    async with asyncio.TaskGroup() as tg:
        for item, task_id in zip(items_list, task_ids):
            tg.create_task(record_label(item, task_id))
            activity.heartbeat()

    return len(task_ids)


def _get_ls_client() -> LabelStudio:
    """Local indirection over `activities.utils.get_ls_client` so unit
    tests can monkeypatch a single symbol on `populate_utils` and have
    both `create_or_get_label_studio_project` and
    `import_tasks_and_record_labels` pick up the fake client."""
    return get_ls_client()
