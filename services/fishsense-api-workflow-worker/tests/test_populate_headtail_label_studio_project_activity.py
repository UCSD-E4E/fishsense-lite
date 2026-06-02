"""Unit tests for populate_headtail_label_studio_project_activity.

Two correctness invariants particular to stage 5.3:
  * Only images carrying a *valid* LaserLabel (completed=True,
    superseded=False, both x/y populated) are candidates — laser
    labeling + the validator have signed off on these. Anything
    weaker isn't usable downstream.
  * The `superseded` cleanup pass marks pre-existing incomplete
    headtail rows as obsolete after a re-import, so downstream
    measurement reads only the freshest row per image.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.headtail_label import HeadTailLabel
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_workflow_worker.activities import (
    populate_headtail_label_studio_project_activity as sut,
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


def _laser(
    image_id: int,
    *,
    completed: bool = True,
    superseded: bool = False,
    x: Optional[float] = 100.0,
    y: Optional[float] = 200.0,
) -> LaserLabel:
    return LaserLabel(
        id=image_id * 7,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=43,
        x=x,
        y=y,
        label="laser",
        updated_at=None,
        superseded=superseded,
        completed=completed,
        label_studio_json={},
        image_id=image_id,
        user_id=None,
    )


def _headtail_label(
    image_id: int,
    *,
    completed: bool,
    superseded: bool = False,
    has_id: bool = True,
) -> HeadTailLabel:
    return HeadTailLabel(
        id=image_id * 1000 if has_id else None,
        label_studio_task_id=image_id * 11,
        label_studio_project_id=71,
        head_x=None,
        head_y=None,
        tail_x=None,
        tail_y=None,
        updated_at=None,
        superseded=superseded,
        completed=completed,
        label_studio_json={},
        image_id=image_id,
        user_id=None,
    )


def test_select_targets_filters_by_valid_laser_and_drops_completed():
    laser = [
        _laser(1),                             # valid + completed headtail -> drop
        _laser(2, completed=False),            # incomplete laser -> drop
        _laser(3),                             # valid + no headtail -> keep
        _laser(4, superseded=True),            # superseded laser -> drop
        _laser(5, x=None),                     # null x -> drop
        _laser(6, y=None),                     # null y -> drop
    ]
    images_by_id = {
        1: _image(1, "a"),
        2: _image(2, "b"),
        3: _image(3, "c"),
        4: _image(4, "d"),
        5: _image(5, "e"),
        6: _image(6, "f"),
    }
    existing = [_headtail_label(1, completed=True)]

    selected = sut._select_target_images(laser, images_by_id, existing)  # pylint: disable=protected-access

    assert [img.id for img in selected] == [3]


def test_build_task_emits_dual_image_and_img_keys(monkeypatch):
    """Pinned: dual-key `image` + `img` shape for legacy LS project
    XML compatibility — see laser populate test of the same name.
    Reverting either key would re-introduce the populate regression
    observed on 2026-05-03."""
    monkeypatch.setenv("E4EFS_OBJECT_STORE__BUCKET", "fishsense-test")
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    expected_url = "s3://fishsense-test/preprocess_headtail_jpeg/abc123.JPG"
    task = sut._build_task(_image(7, "abc123"))  # pylint: disable=protected-access

    assert task["data"] == {"image": expected_url, "img": expected_url}
    assert not task["annotations"]
    assert not task["predictions"]


def _make_fs_client(
    laser_labels: List[LaserLabel],
    existing_headtail: List[HeadTailLabel],
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
    fs.labels.get_laser_labels = AsyncMock(return_value=laser_labels)
    fs.labels.get_headtail_labels = AsyncMock(return_value=existing_headtail)
    fs.labels.put_headtail_label = AsyncMock()
    return fs


def _make_ls_client(returned_task_ids: List[int]):
    ls = MagicMock()
    ls.projects = MagicMock()
    ls.projects.import_tasks = MagicMock(
        return_value=SimpleNamespace(task_ids=returned_task_ids)
    )
    return ls


@pytest.mark.asyncio
async def test_imports_targets_and_supersedes_incomplete_old_rows(monkeypatch):
    """Image 1 has a completed old row -> skip. Image 2 has an
    incomplete old row with id -> get a new task AND old row gets
    superseded. Image 3 is fresh -> get a new task. Image 4 has an
    incomplete old row but no `id` -> superseded skipped (can't
    update without an id) but the new task still goes through."""
    laser = [
        _laser(1),
        _laser(2),
        _laser(3),
        _laser(4),
    ]
    images_by_id = {
        1: _image(1, "a"),
        2: _image(2, "b"),
        3: _image(3, "c"),
        4: _image(4, "d"),
    }
    existing = [
        _headtail_label(1, completed=True),
        _headtail_label(2, completed=False, has_id=True),
        _headtail_label(4, completed=False, has_id=False),
    ]

    fs = _make_fs_client(laser, existing, images_by_id)
    ls = _make_ls_client(returned_task_ids=[3001, 3002, 3003])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    n = await ActivityEnvironment().run(
        sut.populate_headtail_label_studio_project_activity, 42, 71
    )

    assert n == 3

    written = [c.args[1] for c in fs.labels.put_headtail_label.await_args_list]
    new_writes = [w for w in written if w.id is None]
    superseded_writes = [w for w in written if w.id is not None and w.superseded]

    assert {w.image_id for w in new_writes} == {2, 3, 4}
    assert {w.image_id for w in superseded_writes} == {2}


@pytest.mark.asyncio
async def test_no_valid_laser_targets_skips_import_but_still_supersedes(monkeypatch):
    """Edge: dive has incomplete old rows but no laser-valid images.
    Don't push tasks, but DO supersede the stale rows so they don't
    linger as canonical."""
    laser = [_laser(1, completed=False)]
    images_by_id = {1: _image(1, "a")}
    existing = [_headtail_label(1, completed=False, has_id=True)]

    fs = _make_fs_client(laser, existing, images_by_id)
    ls = _make_ls_client(returned_task_ids=[])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    n = await ActivityEnvironment().run(
        sut.populate_headtail_label_studio_project_activity, 42, 71
    )

    assert n == 0
    ls.projects.import_tasks.assert_not_called()
    fs.labels.put_headtail_label.assert_awaited_once()
    written = fs.labels.put_headtail_label.await_args.args[1]
    assert written.superseded is True
