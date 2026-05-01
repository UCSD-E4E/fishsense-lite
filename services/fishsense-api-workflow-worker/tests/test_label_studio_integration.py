"""End-to-end integration tests against the local devcontainer's
Label Studio instance.

These tests close the gap that mocks fundamentally can't: the LS REST
API contract. We exercise the create-side workflow and the populate-
side activity against a real LS server, asserting both the round-trip
shape (project IDs, task IDs) and the silent-corruption-class
invariants — the most important being that
`ls.projects.import_tasks(...).task_ids` comes back in input order, an
assumption the populate activity relies on for label-row alignment.

The fishsense-api side is still mocked here. Full populate-with-real-
api integration needs SQL seeding scaffolding (Camera, Dive, Image,
existing labels) which is its own block of work — listed as a
follow-up in the wrap-up.

Run with:
    ./check.sh integration
or directly:
    uv run --package fishsense-api-workflow-worker python -m pytest \
        services/fishsense-api-workflow-worker/tests/test_label_studio_integration.py \
        -m integration
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_workflow_worker.activities import (
    create_laser_label_studio_project_activity as create_sut,
    populate_laser_label_studio_project_activity as populate_sut,
    populate_utils,
)

pytestmark = pytest.mark.integration


# Minimal labeling config XML — enough for LS to accept image tasks
# without forcing UI clicks to set up a real labeling interface.
_MIN_LASER_XML = (
    "<View>"
    "<Image name='image' value='$image'/>"
    "<KeyPointLabels name='kp-1' toName='image'>"
    "<Label value='laser' background='red'/>"
    "</KeyPointLabels>"
    "</View>"
)


def _image(image_id: int, checksum: str) -> Image:
    return Image(
        id=image_id,
        path=f"path/{image_id}.ORF",
        taken_datetime=datetime(2024, 8, 21, tzinfo=timezone.utc),
        checksum=checksum,
        is_canonical=True,
        dive_id=1,
        camera_id=6,
    )


# ---------------------------------------------------------------------------
# Create-side: idempotent title-lookup against real LS.
# ---------------------------------------------------------------------------


async def test_create_activity_creates_then_returns_existing_on_rerun(monkeypatch):
    """First run creates the project; second run finds it by title and
    returns the same ID. This is the contract that lets us re-run the
    Create workflow without accumulating duplicate projects."""
    from fishsense_api_workflow_worker.activities.utils import get_ls_client

    import uuid

    title = f"fs-create-rerun-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(create_sut, "LASER_PROJECT_TITLE", title)
    monkeypatch.setattr(create_sut, "LASER_LABELING_CONFIG_XML", _MIN_LASER_XML)

    ls = get_ls_client()
    try:
        first_id = await ActivityEnvironment().run(
            create_sut.create_laser_label_studio_project_activity
        )
        second_id = await ActivityEnvironment().run(
            create_sut.create_laser_label_studio_project_activity
        )

        assert first_id == second_id

        # Confirm exactly one project with this title exists.
        matches = [p for p in ls.projects.list() if p.title == title]
        assert len(matches) == 1
        assert matches[0].id == first_id
    finally:
        try:
            ls.projects.delete(first_id)
        except Exception:  # pylint: disable=broad-except
            pass


async def test_create_activity_raises_when_xml_constant_empty(monkeypatch):
    """Empty XML must raise rather than silently make an unlabel-able
    project, even when LS is reachable."""
    import uuid

    title = f"fs-empty-xml-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(create_sut, "LASER_PROJECT_TITLE", title)
    monkeypatch.setattr(create_sut, "LASER_LABELING_CONFIG_XML", "")

    with pytest.raises(Exception) as exc_info:
        await ActivityEnvironment().run(
            create_sut.create_laser_label_studio_project_activity
        )

    assert "labeling-config XML" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Populate-side: real LS, mocked fishsense-api.
# ---------------------------------------------------------------------------


def _make_fs_client(images: List[Image], existing_labels: List[LaserLabel]):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.images = MagicMock()
    fs.images.get = AsyncMock(return_value=images)
    fs.labels = MagicMock()
    fs.labels.get_laser_labels = AsyncMock(return_value=existing_labels)
    fs.labels.put_laser_label = AsyncMock()
    return fs


async def test_populate_imports_tasks_to_real_ls_in_input_order(
    monkeypatch, label_studio_test_project
):
    """The unmockable correctness invariant: when populate calls
    `ls.projects.import_tasks(request=tasks).task_ids`, the returned IDs
    must align positionally with `tasks` so `zip(items_list, task_ids)`
    in `populate_utils.import_tasks_and_record_labels` produces the
    correct (image, task_id) pairs.

    We verify by:
      1. Pushing N tasks with distinct image-URL checksums via populate.
      2. Reading back the task list from LS by project_id.
      3. Asserting the LS-side `data.image` URL of task index `i`
         matches our input image checksum at index `i`.
    """
    project_id, ls = label_studio_test_project

    images = [_image(i + 1, f"checksum-{i:03d}") for i in range(10)]

    fs = _make_fs_client(images, existing_labels=[])
    monkeypatch.setattr(populate_sut, "get_fs_client", lambda: fs)

    n = await ActivityEnvironment().run(
        populate_sut.populate_laser_label_studio_project_activity,
        42,
        project_id,
    )

    assert n == 10

    # Read tasks back from LS and reconcile against what populate told
    # the api SDK to write.
    ls_tasks = list(ls.tasks.list(project=project_id))
    assert len(ls_tasks) == 10

    # The label rows populate wrote — their (image_id, task_id) pairs
    # must match LS's (data.image, task.id) pairs.
    written_labels = [c.args[1] for c in fs.labels.put_laser_label.await_args_list]
    label_by_task_id = {label.label_studio_task_id: label for label in written_labels}

    for ls_task in ls_tasks:
        assert ls_task.id in label_by_task_id, (
            f"populate wrote no LaserLabel row for LS task {ls_task.id}"
        )
        label = label_by_task_id[ls_task.id]
        # The image URL LS persisted must match the image whose ID populate
        # used to anchor the LaserLabel row. If task IDs came back out of
        # order, label.image_id would be wrong for this URL.
        expected_checksum = next(
            img.checksum for img in images if img.id == label.image_id
        )
        assert expected_checksum in ls_task.data["image"], (
            f"task-id misalignment: LS task {ls_task.id} has image URL "
            f"{ls_task.data['image']!r} but the LaserLabel row pointing at "
            f"that task ID is for image_id={label.image_id} (checksum "
            f"{expected_checksum!r}). zip() ordering assumption is broken."
        )


async def test_populate_skips_completed_against_real_ls(
    monkeypatch, label_studio_test_project
):
    """Re-running populate after one image is marked complete in SQL
    must push tasks for the remaining images only — same idempotency
    contract as the unit test, but verified against a real LS that
    actually persists tasks between activity invocations."""
    project_id, ls = label_studio_test_project

    images = [_image(i + 1, f"recheck-{i:03d}") for i in range(5)]

    fs = _make_fs_client(images, existing_labels=[])
    monkeypatch.setattr(populate_sut, "get_fs_client", lambda: fs)

    # First pass: 5 tasks pushed.
    n1 = await ActivityEnvironment().run(
        populate_sut.populate_laser_label_studio_project_activity,
        42,
        project_id,
    )
    assert n1 == 5

    # Second pass: mark image 1 + 2 completed, re-run.
    completed_labels = [
        LaserLabel(
            id=None,
            label_studio_task_id=99,
            label_studio_project_id=project_id,
            x=None,
            y=None,
            label=None,
            updated_at=None,
            superseded=False,
            completed=True,
            label_studio_json={},
            image_id=img.id,
            user_id=None,
        )
        for img in images[:2]
    ]
    fs2 = _make_fs_client(images, existing_labels=completed_labels)
    monkeypatch.setattr(populate_sut, "get_fs_client", lambda: fs2)

    n2 = await ActivityEnvironment().run(
        populate_sut.populate_laser_label_studio_project_activity,
        42,
        project_id,
    )
    assert n2 == 3  # only the 3 still-incomplete images

    # LS now has 5 + 3 = 8 tasks. (Populate doesn't dedupe via LS — it
    # dedupes against SQL state. Re-running with the same SQL state
    # would push duplicates; that's why the SQL-based completed-filter
    # is the correctness gate.)
    ls_tasks = list(ls.tasks.list(project=project_id))
    assert len(ls_tasks) == 8
