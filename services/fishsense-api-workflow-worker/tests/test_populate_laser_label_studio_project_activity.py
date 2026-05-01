"""Unit tests for populate_laser_label_studio_project_activity.

Both clients are mocked. The activity has two failure modes worth
guarding: (a) silently skipping already-completed images is the
correctness contract — the notebook does this and re-running the
workflow must be safe; (b) LS `import_tasks` returning a different
count than we sent must abort before writing mismatched LaserLabel
rows.
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
    populate_laser_label_studio_project_activity as sut,
    populate_utils as sut_utils,
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


def _label(image_id: int, *, completed: bool) -> LaserLabel:
    return LaserLabel(
        id=None,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=73,
        x=None,
        y=None,
        label=None,
        updated_at=None,
        superseded=False,
        completed=completed,
        label_studio_json={},
        image_id=image_id,
        user_id=None,
    )


def test_select_incomplete_filters_completed():
    images = [_image(1, "a"), _image(2, "b"), _image(3, "c")]
    existing = [_label(1, completed=True), _label(2, completed=False)]

    result = sut._select_incomplete_images(images, existing)  # pylint: disable=protected-access

    assert [img.id for img in result] == [2, 3]


def test_build_task_uses_configured_url_base(monkeypatch):
    monkeypatch.setenv(
        "E4EFS_LABEL_STUDIO__IMAGE_URL_BASE", "https://orchestrator.example.com"
    )
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    task = sut._build_task(_image(7, "abc123"))  # pylint: disable=protected-access

    assert task == {
        "data": {
            "image": "https://orchestrator.example.com/api/v1/data/preprocess_jpeg/abc123"
        },
        "annotations": [],
    }


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


def _make_ls_client(returned_task_ids: List[int]):
    ls = MagicMock()
    ls.projects = MagicMock()
    ls.projects.import_tasks = MagicMock(
        return_value=SimpleNamespace(task_ids=returned_task_ids)
    )
    return ls


@pytest.mark.asyncio
async def test_imports_tasks_and_writes_one_label_per_incomplete_image(
    monkeypatch,
):
    images = [_image(1, "a"), _image(2, "b"), _image(3, "c")]
    existing = [_label(1, completed=True)]  # image 1 already done

    fs = _make_fs_client(images, existing)
    ls = _make_ls_client(returned_task_ids=[1001, 1002])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    n = await ActivityEnvironment().run(
        sut.populate_laser_label_studio_project_activity, 42, 73
    )

    assert n == 2
    ls.projects.import_tasks.assert_called_once()
    args, kwargs = ls.projects.import_tasks.call_args
    assert args[0] == 73
    assert kwargs["return_task_ids"] is True
    assert len(kwargs["request"]) == 2
    assert fs.labels.put_laser_label.await_count == 2

    written_labels = [c.args[1] for c in fs.labels.put_laser_label.await_args_list]
    assert {label.image_id for label in written_labels} == {2, 3}
    assert {label.label_studio_task_id for label in written_labels} == {1001, 1002}
    assert all(label.label_studio_project_id == 73 for label in written_labels)
    assert all(label.completed is False for label in written_labels)


@pytest.mark.asyncio
async def test_no_incomplete_images_is_a_no_op(monkeypatch):
    images = [_image(1, "a")]
    existing = [_label(1, completed=True)]

    fs = _make_fs_client(images, existing)
    ls = _make_ls_client(returned_task_ids=[])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    n = await ActivityEnvironment().run(
        sut.populate_laser_label_studio_project_activity, 42, 73
    )

    assert n == 0
    ls.projects.import_tasks.assert_not_called()
    fs.labels.put_laser_label.assert_not_called()


@pytest.mark.asyncio
async def test_aborts_when_import_tasks_returns_wrong_count(monkeypatch):
    images = [_image(1, "a"), _image(2, "b")]
    fs = _make_fs_client(images, existing_labels=[])
    ls = _make_ls_client(returned_task_ids=[999])  # only 1 ID for 2 tasks

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    with pytest.raises(Exception):
        await ActivityEnvironment().run(
            sut.populate_laser_label_studio_project_activity, 42, 73
        )

    fs.labels.put_laser_label.assert_not_called()
