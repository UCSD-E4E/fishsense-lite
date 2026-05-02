"""Unit tests for populate_species_label_studio_project_activity."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.species_label import SpeciesLabel
from fishsense_api_workflow_worker.activities import (
    populate_species_label_studio_project_activity as sut,
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


def _label(
    image_id: int, *, completed: bool, project_id: int | None = 70
) -> SpeciesLabel:
    return SpeciesLabel(
        id=None,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=project_id,
        image_url=None,
        updated_at=None,
        completed=completed,
        grouping=None,
        top_three_photos_of_group=None,
        slate_upside_down=None,
        laser_x=None,
        laser_y=None,
        laser_label=None,
        content_of_image=None,
        fish_measurable_category=None,
        fish_angle_category=None,
        fish_curved_category=None,
        label_studio_json={},
        image_id=image_id,
        user_id=None,
    )


def test_select_unlabeled_excludes_images_with_any_completed_label():
    images = [_image(1, "a"), _image(2, "b"), _image(3, "c")]
    existing = [_label(1, completed=True), _label(2, completed=False)]

    result = sut._select_unlabeled_images(images, existing)  # pylint: disable=protected-access

    assert [img.id for img in result] == [2, 3]


def test_select_unlabeled_handles_multi_row_state():
    """Same multi-row hardening as the laser populate. An image with
    a completed row in one project plus an incomplete sentinel in
    another is treated as labeled — the previous dict-collapse filter
    could go either way depending on iteration order."""
    images = [_image(1, "a"), _image(2, "b")]
    existing = [
        _label(1, completed=True, project_id=70),
        _label(1, completed=False, project_id=None),
        _label(2, completed=False, project_id=70),
    ]

    result = sut._select_unlabeled_images(images, existing)  # pylint: disable=protected-access

    assert [img.id for img in result] == [2]


def test_build_task_uses_groups_jpeg_folder(monkeypatch):
    monkeypatch.setenv(
        "E4EFS_LABEL_STUDIO__IMAGE_URL_BASE", "https://orchestrator.example.com"
    )
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    task = sut._build_task(_image(7, "abc123"))  # pylint: disable=protected-access

    assert task["data"]["image"] == (
        "https://orchestrator.example.com/api/v1/data/groups_jpeg/abc123"
    )
    assert not task["predictions"]


def _make_fs_client(images: List[Image], existing_labels: List[SpeciesLabel]):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.images = MagicMock()
    fs.images.get = AsyncMock(return_value=images)
    fs.labels = MagicMock()
    fs.labels.get_species_labels = AsyncMock(return_value=existing_labels)
    fs.labels.put_species_label = AsyncMock()
    return fs


def _make_ls_client(returned_task_ids: List[int]):
    ls = MagicMock()
    ls.projects = MagicMock()
    ls.projects.import_tasks = MagicMock(
        return_value=SimpleNamespace(task_ids=returned_task_ids)
    )
    return ls


@pytest.mark.asyncio
async def test_imports_tasks_and_writes_one_label_per_incomplete_image(monkeypatch):
    images = [_image(1, "a"), _image(2, "b"), _image(3, "c")]
    existing = [_label(1, completed=True)]

    fs = _make_fs_client(images, existing)
    ls = _make_ls_client(returned_task_ids=[2001, 2002])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    n = await ActivityEnvironment().run(
        sut.populate_species_label_studio_project_activity, 42, 70
    )

    assert n == 2
    assert fs.labels.put_species_label.await_count == 2
    written = [c.args[1] for c in fs.labels.put_species_label.await_args_list]
    assert {label.image_id for label in written} == {2, 3}
    assert {label.label_studio_task_id for label in written} == {2001, 2002}
    assert all(label.label_studio_project_id == 70 for label in written)
    assert all(label.image_url and "groups_jpeg" in label.image_url for label in written)


@pytest.mark.asyncio
async def test_no_incomplete_images_is_a_no_op(monkeypatch):
    images = [_image(1, "a")]
    existing = [_label(1, completed=True)]

    fs = _make_fs_client(images, existing)
    ls = _make_ls_client(returned_task_ids=[])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    n = await ActivityEnvironment().run(
        sut.populate_species_label_studio_project_activity, 42, 70
    )

    assert n == 0
    ls.projects.import_tasks.assert_not_called()
    fs.labels.put_species_label.assert_not_called()
