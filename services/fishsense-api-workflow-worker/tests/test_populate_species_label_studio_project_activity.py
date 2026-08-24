"""Unit tests for populate_species_label_studio_project_activity.

Correctness invariant particular to species (post 2026-05-05): only
images carrying a *valid* LaserLabel (completed=True, superseded=False,
both x/y populated) are candidates — same gate as head/tail. Cascades
from laser labelers + the validator signing off.

The activity is idempotent so it can run on a schedule: an image that
already has a non-superseded species row for the target project is not
re-imported, and the end-of-run supersede pass only dead-letters
incomplete rows in *other* (stale) projects — never the target
project's own in-progress rows.
"""

from __future__ import annotations


from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_sdk.models.species_label import SpeciesLabel
from fishsense_api_workflow_worker.activities import (

    populate_species_label_studio_project_activity as sut,
    populate_utils as sut_utils,
)

# Shared with the other populate suites — see `worker_tests_support.populate`.
from worker_tests_support.populate import (  # noqa: F401
    images_for,
    laser_label_matrix,
    patch_jpeg_gate,
    image as _image,
    laser_label as _laser,
    fake_label_studio_client as _make_ls_client,
)


def _species_label(
    image_id: int,
    *,
    completed: bool,
    has_id: bool = True,
    project_id: int | None = 70,
) -> SpeciesLabel:
    return SpeciesLabel(
        id=image_id * 1000 if has_id else None,
        label_studio_task_id=image_id * 11,
        label_studio_project_id=project_id,
        image_url=None,
        updated_at=None,
        completed=completed,
        superseded=False,
        label_studio_json={},
        image_id=image_id,
        user_id=None,
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
    )


@pytest.fixture(autouse=True)
def _all_jpegs_present(monkeypatch):
    """Default the JPEG gate to "present" so activity tests exercise the
    import path; the gate test overrides this with a selective fake."""
    return patch_jpeg_gate(monkeypatch, sut)


def test_select_targets_filters_by_valid_laser_and_drops_completed():
    laser = laser_label_matrix()
    images_by_id = images_for()
    existing = [_species_label(1, completed=True)]

    selected = sut._select_target_images(laser, images_by_id, existing, 70)  # pylint: disable=protected-access

    assert [img.id for img in selected] == [3]


def test_select_targets_skips_images_already_in_this_project():
    """Idempotency filter: an image with a non-superseded species row for
    the *target* project is not re-selected, but one whose only row is in
    a different (stale) project still is."""
    laser = [_laser(1), _laser(2)]
    images_by_id = {1: _image(1, "a"), 2: _image(2, "b")}
    existing = [
        _species_label(1, completed=False, project_id=70),   # already in target -> skip
        _species_label(2, completed=False, project_id=99),   # stale old project -> keep
    ]

    selected = sut._select_target_images(laser, images_by_id, existing, 70)  # pylint: disable=protected-access

    assert [img.id for img in selected] == [2]


def test_build_task_uses_groups_jpeg_folder_and_dual_keys(monkeypatch):
    """Pinned: dual-key `image` + `img` shape — see the laser populate
    test of the same name for the rationale (legacy LS project XML
    uses both conventions; emitting one fails import_tasks)."""
    monkeypatch.setenv("E4EFS_OBJECT_STORE__BUCKET", "fishsense-test")
    from fishsense_api_workflow_worker import config as cfg
    cfg.settings.reload()

    # Physical Garage prefix (preprocess_groups_jpeg) + `.JPG`; LS
    # presigns this `s3://` URI.
    expected_url = "s3://fishsense-test/preprocess_groups_jpeg/abc123.JPG"
    task = sut._build_task(_image(7, "abc123"))  # pylint: disable=protected-access

    assert task["data"]["image"] == expected_url
    assert task["data"]["img"] == expected_url
    assert not task["predictions"]


def _make_fs_client(
    laser_labels: List[LaserLabel],
    existing_species: List[SpeciesLabel],
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
    fs.labels.get_species_labels = AsyncMock(return_value=existing_species)
    fs.labels.put_species_label = AsyncMock()
    return fs


@pytest.mark.asyncio
async def test_imports_targets_and_writes_new_labels(monkeypatch):
    """Migration path onto a fresh project (70), with stale rows in an
    old project (99). Image 1 has a completed old row -> skip. Image 2
    has an incomplete stale-project row with id -> new task + the stale
    row is superseded. Image 3 is fresh -> new task. Image 4 has an
    incomplete stale-project row but no id -> new task, stale row left
    (can't supersede an unpersisted row)."""
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
        _species_label(1, completed=True, project_id=99),
        _species_label(2, completed=False, has_id=True, project_id=99),
        _species_label(4, completed=False, has_id=False, project_id=99),
    ]

    fs = _make_fs_client(laser, existing, images_by_id)
    ls = _make_ls_client(returned_task_ids=[3001, 3002, 3003])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    n = await ActivityEnvironment().run(
        sut.populate_species_label_studio_project_activity, 42, 70
    )

    assert n == 3

    written = [c.args[1] for c in fs.labels.put_species_label.await_args_list]
    # Fresh task rows for the laser-valid, not-yet-completed images.
    new_writes = [w for w in written if w.id is None]
    assert {w.image_id for w in new_writes} == {2, 3, 4}
    # Supersede pass retires the pre-existing incomplete-with-id row (image 2);
    # the completed row (image 1) and the id-less row (image 4) are untouched.
    superseded_writes = [w for w in written if w.superseded]
    assert {w.image_id for w in superseded_writes} == {2}


@pytest.mark.asyncio
async def test_no_valid_laser_targets_skips_import_but_supersedes_stale(monkeypatch):
    """No laser-valid images -> no task import, but the supersede pass still
    retires a pre-existing incomplete-with-id species row in a stale
    (different) project."""
    laser = [_laser(1, completed=False)]
    images_by_id = {1: _image(1, "a")}
    existing = [_species_label(1, completed=False, has_id=True, project_id=99)]

    fs = _make_fs_client(laser, existing, images_by_id)
    ls = _make_ls_client(returned_task_ids=[])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    n = await ActivityEnvironment().run(
        sut.populate_species_label_studio_project_activity, 42, 70
    )

    assert n == 0
    ls.projects.import_tasks.assert_not_called()
    written = [c.args[1] for c in fs.labels.put_species_label.await_args_list]
    assert [(w.image_id, w.superseded) for w in written] == [(1, True)]


@pytest.mark.asyncio
async def test_defers_images_whose_jpeg_is_not_in_garage(monkeypatch):
    """JPEG gate: an image whose species JPEG isn't in Garage yet is not
    imported (deferred to a later run), so a scheduled populate never
    seeds a species row ahead of preprocess writing the JPEG — which
    would strand the image outside the preprocess cohort."""
    laser = [_laser(1), _laser(2)]
    images_by_id = {1: _image(1, "aaa"), 2: _image(2, "bbb")}
    existing: List[SpeciesLabel] = []

    fs = _make_fs_client(laser, existing, images_by_id)
    ls = _make_ls_client(returned_task_ids=[9001])  # only image 1 imports

    store = MagicMock()

    async def _has(_folder, checksum):
        return checksum == "aaa"  # image 2's JPEG (bbb) is missing

    store.has_processed_jpeg = AsyncMock(side_effect=_has)

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(sut, "open_object_store_client", lambda: store)

    n = await ActivityEnvironment().run(
        sut.populate_species_label_studio_project_activity, 42, 70
    )

    assert n == 1
    written = [c.args[1] for c in fs.labels.put_species_label.await_args_list]
    new_writes = [w for w in written if w.id is None]
    assert {w.image_id for w in new_writes} == {1}  # image 2 deferred


@pytest.mark.asyncio
async def test_rerun_is_idempotent_for_same_project(monkeypatch):
    """Scheduling invariant: a re-run where every laser-valid image
    already has a non-superseded task row *for this project* imports
    nothing and supersedes nothing — so the activity can be put on a
    schedule without churning duplicate LS tasks."""
    laser = [_laser(1), _laser(2)]
    images_by_id = {1: _image(1, "a"), 2: _image(2, "b")}
    # both already have a live (non-superseded, incomplete) task in project 70
    existing = [
        _species_label(1, completed=False, project_id=70),
        _species_label(2, completed=False, project_id=70),
    ]

    fs = _make_fs_client(laser, existing, images_by_id)
    ls = _make_ls_client(returned_task_ids=[])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    n = await ActivityEnvironment().run(
        sut.populate_species_label_studio_project_activity, 42, 70
    )

    assert n == 0
    ls.projects.import_tasks.assert_not_called()
    # the project's own in-progress rows are left untouched (no supersede churn)
    assert not fs.labels.put_species_label.await_args_list


@pytest.mark.asyncio
async def test_writes_label_with_image_url_and_groups_jpeg_folder(monkeypatch):
    """The post-2026-05-05 species populate writes a label whose
    image_url uses the species JPEG prefix, matching the LS task
    URL — so downstream sync can recover the JPEG from the row."""
    monkeypatch.setenv("E4EFS_OBJECT_STORE__BUCKET", "fishsense-test")
    from fishsense_api_workflow_worker import config as cfg
    cfg.settings.reload()

    laser = [_laser(1)]
    images_by_id = {1: _image(1, "abc123")}
    existing = []

    fs = _make_fs_client(laser, existing, images_by_id)
    ls = _make_ls_client(returned_task_ids=[5001])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.populate_species_label_studio_project_activity, 42, 70
    )

    written = fs.labels.put_species_label.await_args.args[1]
    assert written.image_id == 1
    assert written.label_studio_task_id == 5001
    assert written.label_studio_project_id == 70
    assert written.image_url is not None
    assert "preprocess_groups_jpeg" in written.image_url
    assert "abc123" in written.image_url


@pytest.mark.asyncio
async def test_publishes_when_no_images_deferred(monkeypatch):
    # All laser-valid images have their JPEG (autouse fixture) -> nothing
    # deferred -> project task set complete -> publish.
    laser = [_laser(1), _laser(2)]
    images_by_id = {1: _image(1, "a"), 2: _image(2, "b")}
    fs = _make_fs_client(laser, [], images_by_id)
    ls = _make_ls_client(returned_task_ids=[1, 2])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.populate_species_label_studio_project_activity, 42, 70
    )

    ls.projects.update.assert_called_once_with(id=70, is_published=True)


@pytest.mark.asyncio
async def test_does_not_publish_when_an_image_is_deferred(monkeypatch):
    # Image 2's JPEG isn't in Garage yet -> deferred -> project incomplete ->
    # stay a hidden draft even though image 1's task imported.
    laser = [_laser(1), _laser(2)]
    images_by_id = {1: _image(1, "aaa"), 2: _image(2, "bbb")}
    fs = _make_fs_client(laser, [], images_by_id)
    ls = _make_ls_client(returned_task_ids=[9001])

    store = MagicMock()

    async def _has(_folder, checksum):
        return checksum == "aaa"

    store.has_processed_jpeg = AsyncMock(side_effect=_has)

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(sut, "open_object_store_client", lambda: store)

    await ActivityEnvironment().run(
        sut.populate_species_label_studio_project_activity, 42, 70
    )

    ls.projects.update.assert_not_called()


@pytest.mark.asyncio
async def test_does_not_publish_empty_project(monkeypatch):
    # No laser-valid images and no existing rows -> nothing to task ->
    # stay a hidden draft.
    laser = [_laser(1, completed=False)]
    images_by_id = {1: _image(1, "a")}
    fs = _make_fs_client(laser, [], images_by_id)
    ls = _make_ls_client(returned_task_ids=[])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.populate_species_label_studio_project_activity, 42, 70
    )

    ls.projects.update.assert_not_called()
