"""Unit tests for populate_headtail_label_studio_project_activity.

Two correctness invariants particular to stage 5.3:
  * Only species labels with `top_three_photos_of_group=True` are
    candidates — that flag is the species labeler's hand-pick of
    measurable angles. Pushing every species-labeled image would
    flood the headtail project.
  * The `superseded` cleanup pass marks pre-existing incomplete
    headtail rows as obsolete after a re-import, so downstream
    measurement reads only the freshest row per image.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.headtail_label import HeadTailLabel
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.species_label import SpeciesLabel
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


def _species_label(image_id: int, *, top_three: bool) -> SpeciesLabel:
    return SpeciesLabel(
        id=image_id * 100,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=70,
        image_url=None,
        updated_at=None,
        completed=True,
        grouping=None,
        top_three_photos_of_group=top_three,
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


def test_select_targets_filters_by_top_three_and_drops_completed():
    species = [
        _species_label(1, top_three=True),
        _species_label(2, top_three=False),
        _species_label(3, top_three=True),
    ]
    images_by_id = {1: _image(1, "a"), 3: _image(3, "c")}
    existing = [_headtail_label(1, completed=True)]

    selected = sut._select_target_images(species, images_by_id, existing)

    assert [img.id for img in selected] == [3]


def _make_fs_client(
    species_labels: List[SpeciesLabel],
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
    fs.labels.get_species_labels = AsyncMock(return_value=species_labels)
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
    species = [
        _species_label(1, top_three=True),
        _species_label(2, top_three=True),
        _species_label(3, top_three=True),
        _species_label(4, top_three=True),
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

    fs = _make_fs_client(species, existing, images_by_id)
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
async def test_no_top_three_targets_skips_import_but_still_supersedes(monkeypatch):
    """Edge: dive has incomplete old rows but no top_three species
    flagged. Don't push tasks, but DO supersede the stale rows so
    they don't linger as canonical."""
    species = [_species_label(1, top_three=False)]
    images_by_id = {1: _image(1, "a")}
    existing = [_headtail_label(1, completed=False, has_id=True)]

    fs = _make_fs_client(species, existing, images_by_id)
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
