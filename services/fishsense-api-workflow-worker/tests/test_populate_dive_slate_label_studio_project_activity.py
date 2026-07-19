"""Unit tests for populate_dive_slate_label_studio_project_activity."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.species_label import SpeciesLabel
from fishsense_api_workflow_worker.activities import (
    populate_dive_slate_label_studio_project_activity as sut,
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


def _species_label(image_id: int, *, content: str | None) -> SpeciesLabel:
    return SpeciesLabel(
        id=image_id * 100,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=70,
        image_url=None,
        updated_at=None,
        completed=True,
        superseded=False,
        grouping=None,
        top_three_photos_of_group=None,
        slate_upside_down=None,
        laser_x=None,
        laser_y=None,
        laser_label=None,
        content_of_image=content,
        fish_measurable_category=None,
        fish_angle_category=None,
        fish_curved_category=None,
        label_studio_json={},
        image_id=image_id,
        user_id=None,
    )


def _slate_label(image_id: int, *, completed: bool) -> DiveSlateLabel:
    return DiveSlateLabel(
        id=image_id * 100,
        label_studio_task_id=image_id * 11,
        label_studio_project_id=66,
        image_url=None,
        upside_down=None,
        reference_points=None,
        slate_rectangle=None,
        skipped_points=None,
        updated_at=None,
        completed=completed,
        superseded=False,
        label_studio_json={},
        image_id=image_id,
        user_id=None,
    )


def test_select_targets_filters_by_slate_marker_and_completion():
    species = [
        _species_label(1, content=sut.SLATE_CONTENT_MARKER),
        _species_label(2, content="Fish"),
        _species_label(3, content=sut.SLATE_CONTENT_MARKER),
    ]
    existing = [_slate_label(1, completed=True)]

    result = sut._select_target_image_ids(species, existing)  # pylint: disable=protected-access

    assert result == [3]


def test_build_task_emits_dual_image_and_img_keys(monkeypatch):
    """Pinned: dual-key `image` + `img` shape for legacy LS project
    XML compatibility — see laser populate test of the same name."""
    monkeypatch.setenv("E4EFS_OBJECT_STORE__BUCKET", "fishsense-test")
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    expected_url = "s3://fishsense-test/preprocess_slate_images_jpeg/abc123.JPG"
    task = sut._build_task(_image(7, "abc123"))  # pylint: disable=protected-access

    assert task["data"] == {"image": expected_url, "img": expected_url}
    assert not task["annotations"]
    assert not task["predictions"]


def _make_fs_client(
    species_labels: List[SpeciesLabel],
    existing_slate: List[DiveSlateLabel],
    images_by_id: dict,
):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    async def _get_image(image_id: int):
        return images_by_id.get(image_id)

    fs.images = MagicMock()
    fs.images.get = AsyncMock(side_effect=_get_image)

    fs.labels = MagicMock()
    fs.labels.get_species_labels = AsyncMock(return_value=species_labels)
    fs.labels.get_dive_slate_labels = AsyncMock(return_value=existing_slate)
    fs.labels.put_dive_slate_label = AsyncMock()
    return fs


def _make_ls_client(returned_task_ids: List[int]):
    ls = MagicMock()
    ls.projects = MagicMock()
    ls.projects.import_tasks = MagicMock(
        return_value=SimpleNamespace(task_ids=returned_task_ids)
    )
    return ls


@pytest.mark.asyncio
async def test_imports_only_slate_marked_images(monkeypatch):
    species = [
        _species_label(1, content=sut.SLATE_CONTENT_MARKER),
        _species_label(2, content="Fish"),
        _species_label(3, content=sut.SLATE_CONTENT_MARKER),
    ]
    images_by_id = {1: _image(1, "a"), 3: _image(3, "c")}

    fs = _make_fs_client(species, existing_slate=[], images_by_id=images_by_id)
    ls = _make_ls_client(returned_task_ids=[4001, 4002])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    n = await ActivityEnvironment().run(
        sut.populate_dive_slate_label_studio_project_activity, 42, 66
    )

    assert n == 2
    written = [c.args[1] for c in fs.labels.put_dive_slate_label.await_args_list]
    assert {label.image_id for label in written} == {1, 3}
    assert all(label.label_studio_project_id == 66 for label in written)
    assert all(
        "preprocess_slate_images_jpeg" in label.image_url for label in written
    )


@pytest.mark.asyncio
async def test_no_slate_marked_images_is_a_no_op(monkeypatch):
    species = [_species_label(1, content="Fish")]
    fs = _make_fs_client(species, existing_slate=[], images_by_id={})
    ls = _make_ls_client(returned_task_ids=[])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    n = await ActivityEnvironment().run(
        sut.populate_dive_slate_label_studio_project_activity, 42, 66
    )

    assert n == 0
    ls.projects.import_tasks.assert_not_called()
    fs.labels.put_dive_slate_label.assert_not_called()


@pytest.mark.asyncio
async def test_publishes_project_after_import(monkeypatch):
    # Slate imports its whole selection in one pass -> project complete
    # after import -> publish.
    species = [
        _species_label(1, content=sut.SLATE_CONTENT_MARKER),
        _species_label(3, content=sut.SLATE_CONTENT_MARKER),
    ]
    images_by_id = {1: _image(1, "a"), 3: _image(3, "c")}
    fs = _make_fs_client(species, existing_slate=[], images_by_id=images_by_id)
    ls = _make_ls_client(returned_task_ids=[4001, 4002])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.populate_dive_slate_label_studio_project_activity, 42, 66
    )

    ls.projects.update.assert_called_once_with(id=66, is_published=True)


@pytest.mark.asyncio
async def test_does_not_publish_empty_project(monkeypatch):
    # No slate-marked images and no existing rows -> stay a hidden draft.
    species = [_species_label(1, content="Fish")]
    fs = _make_fs_client(species, existing_slate=[], images_by_id={})
    ls = _make_ls_client(returned_task_ids=[])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.populate_dive_slate_label_studio_project_activity, 42, 66
    )

    ls.projects.update.assert_not_called()
