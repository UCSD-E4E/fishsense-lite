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


from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.headtail_label import HeadTailLabel
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_workflow_worker.activities import (

    populate_headtail_label_studio_project_activity as sut,
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


def _headtail_label(
    image_id: int,
    *,
    completed: bool,
    superseded: bool = False,
    has_id: bool = True,
    project_id: int = 71,
) -> HeadTailLabel:
    return HeadTailLabel(
        id=image_id * 1000 if has_id else None,
        label_studio_task_id=image_id * 11,
        label_studio_project_id=project_id,
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


@pytest.fixture(autouse=True)
def _all_jpegs_present(monkeypatch):
    """Default the JPEG gate to "present" so activity tests exercise the
    import path; the gate test overrides this with a selective fake.
    Mirrors the species populate test harness."""
    return patch_jpeg_gate(monkeypatch, sut)


def test_select_targets_filters_by_valid_laser_and_drops_completed():
    laser = laser_label_matrix()
    images_by_id = images_for()
    existing = [_headtail_label(1, completed=True)]

    selected = sut._select_target_images(laser, images_by_id, existing)  # pylint: disable=protected-access

    assert [img.id for img in selected] == [3]


def test_build_task_emits_dual_image_and_img_keys(monkeypatch):
    """Pinned: dual-key `image` + `img` shape for legacy LS project
    XML compatibility — see laser populate test of the same name.
    Reverting either key would re-introduce the populate regression
    observed on 2026-05-03."""
    monkeypatch.setenv("E4EFS_OBJECT_STORE__BUCKET", "fishsense-test")
    from fishsense_api_workflow_worker import config as cfg
    cfg.settings.reload()

    expected_url = "s3://fishsense-test/preprocess_headtail_jpeg/abc123.JPG"
    task = sut._build_task(_image(7, "abc123"))  # pylint: disable=protected-access

    assert task["data"] == {"image": expected_url, "img": expected_url}
    assert not task["annotations"]
    assert not task["predictions"]


class _StubPrediction:
    """Minimal `HeadTailPrediction` stand-in.

    Defaults to an *abstention*: it passes populate's prediction gate (the
    detector visited the image) while seeding no keypoints, so these tests keep
    asserting task import and supersede behaviour rather than annotation
    content. Annotation content has its own file.
    """

    def __init__(self, image_id, status="no_detections"):
        self.image_id = image_id
        self.status = status
        self.head_x = self.head_y = self.tail_x = self.tail_y = None
        self.width = self.height = None
        self.silhouette_ratio = None
        self.rejected_low_confidence = False
        self.checkpoint = None
        self.core_version = None


def _make_fs_client(
    laser_labels: List[LaserLabel],
    existing_headtail: List[HeadTailLabel],
    images_by_id: dict,
    predictions=None,
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
    # Populate is prediction-gated, so by default every candidate counts as
    # already visited; a test that cares passes its own list.
    if predictions is None:
        predictions = [_StubPrediction(image_id) for image_id in images_by_id]
    fs.labels.get_headtail_predictions = AsyncMock(return_value=predictions)
    return fs


@pytest.mark.asyncio
async def test_imports_targets_and_supersedes_incomplete_old_rows(monkeypatch):
    """Image 1 has a completed old row -> skip. Image 2 has an incomplete old
    row with id IN THIS PROJECT -> re-imported, and must NOT be superseded
    (see the flip-flop regression below). Image 3 is fresh -> new task.
    Image 4 has an incomplete old row but no `id` -> superseded skipped
    (can't update without an id) but the new task still goes through."""
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
    assert not superseded_writes, (
        "rows in the project being populated were just refreshed by the "
        "import; superseding them undoes this run's own work"
    )


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


@pytest.mark.asyncio
async def test_publishes_project_after_import(monkeypatch):
    # Headtail imports its whole selection in one pass -> project complete
    # after import -> publish.
    laser = [_laser(1), _laser(2)]
    images_by_id = {1: _image(1, "a"), 2: _image(2, "b")}
    fs = _make_fs_client(laser, [], images_by_id)
    ls = _make_ls_client(returned_task_ids=[3001, 3002])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.populate_headtail_label_studio_project_activity, 42, 71
    )

    ls.projects.update.assert_called_once_with(id=71, is_published=True)


@pytest.mark.asyncio
async def test_does_not_publish_empty_project(monkeypatch):
    # No laser-valid images and no existing rows -> stay a hidden draft.
    laser = [_laser(1, completed=False)]
    images_by_id = {1: _image(1, "a")}
    fs = _make_fs_client(laser, [], images_by_id)
    ls = _make_ls_client(returned_task_ids=[])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.populate_headtail_label_studio_project_activity, 42, 71
    )

    ls.projects.update.assert_not_called()


@pytest.mark.asyncio
async def test_running_twice_does_not_flip_rows_back_to_superseded(monkeypatch):
    """The prod flip-flop (dive 341, 2026-08-04).

    `put_headtail_label` upserts on `image_id`, so there is only ever ONE row
    per image — meaning the "old" row the supersede pass retires IS the row the
    import just created. The pass reads a snapshot taken BEFORE the import and
    skips rows already superseded, so the outcome ALTERNATES between runs:
    superseded -> live -> superseded -> ... Each hourly firing (and each
    activity retry) toggled it, so the dive oscillated in and out of the
    stage-5.1 cohort and its labeler tasks flickered between live and
    dead-lettered.

    Second run must be a no-op on the row state. Mirrors the guard species
    populate already has (`old.label_studio_project_id == project_id`).
    """
    laser = [_laser(1)]
    images_by_id = {1: _image(1, "a")}
    # State after a first populate: a live, incomplete row in THIS project.
    existing = [_headtail_label(1, completed=False, has_id=True)]

    fs = _make_fs_client(laser, existing, images_by_id)
    ls = _make_ls_client(returned_task_ids=[4001])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.populate_headtail_label_studio_project_activity, 42, 71
    )

    written = [c.args[1] for c in fs.labels.put_headtail_label.await_args_list]
    assert not [w for w in written if w.superseded], (
        "a re-run must leave the pending row live, or the dive never drains"
    )


@pytest.mark.asyncio
async def test_stale_rows_for_non_target_images_are_still_superseded(monkeypatch):
    """The supersede pass keeps its job. Image 2's laser is no longer valid, so
    it is NOT re-imported this run — its lingering incomplete row is genuinely
    stale and must be dead-lettered. Only images the import just refreshed are
    exempt."""
    laser = [_laser(1), _laser(2, superseded=True)]
    images_by_id = {1: _image(1, "a"), 2: _image(2, "b")}
    existing = [
        _headtail_label(1, completed=False, has_id=True),  # target -> exempt
        _headtail_label(2, completed=False, has_id=True),  # not a target -> stale
    ]

    fs = _make_fs_client(laser, existing, images_by_id)
    ls = _make_ls_client(returned_task_ids=[4002])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.populate_headtail_label_studio_project_activity, 42, 71
    )

    written = [c.args[1] for c in fs.labels.put_headtail_label.await_args_list]
    superseded = [w for w in written if w.id is not None and w.superseded]
    assert {w.image_id for w in superseded} == {2}


@pytest.mark.asyncio
async def test_defers_images_whose_jpeg_is_not_in_garage(monkeypatch):
    """Never seed a task for an image the data-worker hasn't rendered yet.

    A task whose `s3://` URI has no object behind it shows the labeler a
    missing image, and — because the new row is non-sentinel — it also drops
    the dive out of the stage-5.1 cohort, so the JPEG can never be rendered.
    Prod dive 84 landed in exactly that state on 2026-08-04 (36 of 39 tasks
    pointed at nothing) after populate was run standalone to unblock it.
    Species populate has gated on JPEG presence for this reason; headtail
    didn't. Deferring is safe: the image returns on a later run.
    """
    laser = [_laser(1), _laser(2)]
    images_by_id = {1: _image(1, "a"), 2: _image(2, "b")}

    fs = _make_fs_client(laser, [], images_by_id)
    ls = _make_ls_client(returned_task_ids=[5001])

    # Only image 1's JPEG exists.
    class _Store:
        async def has_processed_jpeg(self, folder, checksum):
            assert folder == sut.HEADTAIL_FOLDER
            return checksum == "a"

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)
    store = _Store()
    monkeypatch.setattr(sut, "open_object_store_client", lambda: store)

    n = await ActivityEnvironment().run(
        sut.populate_headtail_label_studio_project_activity, 42, 71
    )

    assert n == 1, "only the image with a rendered JPEG is seeded"
    written = [c.args[1] for c in fs.labels.put_headtail_label.await_args_list]
    assert {w.image_id for w in written} == {1}


@pytest.mark.asyncio
async def test_legacy_other_project_rows_are_superseded_even_when_refreshed(monkeypatch):
    """A legacy-project row must be dead-lettered even if this run re-imported
    its image.

    `get_headtail_labels(dive_id)` returns every non-superseded row for the
    dive across ALL projects, and `put_headtail_label` upserts on
    `(image_id, label_studio_project_id)` — so one image really can hold both a
    legacy row and this project's row. The exemption used to be image-based
    ("this run refreshed image 1, so skip everything for image 1"), which let
    the legacy row survive forever. Since `dive_pipeline_status`
    `headtail_labeling_complete` requires ZERO incomplete non-superseded rows,
    that dive read incomplete on the dashboard permanently, even after
    labelers finished the per-dive project.

    The exemption is now (same project AND refreshed), so this run's own row
    stays live while the legacy one retires.
    """
    laser = [_laser(1)]
    images_by_id = {1: _image(1, "a")}
    existing = [
        # This project's live row for image 1 — must stay live (dive 341).
        _headtail_label(1, completed=False, has_id=True, project_id=71),
        # A legacy shared-project row for the SAME image — must be superseded.
        _headtail_label(1, completed=False, has_id=True, project_id=66),
    ]

    fs = _make_fs_client(laser, existing, images_by_id)
    ls = _make_ls_client(returned_task_ids=[4003])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.populate_headtail_label_studio_project_activity, 42, 71
    )

    written = [c.args[1] for c in fs.labels.put_headtail_label.await_args_list]
    superseded = [w for w in written if w.superseded]
    assert [w.label_studio_project_id for w in superseded] == [66], (
        "only the legacy-project row should be dead-lettered"
    )


@pytest.mark.asyncio
async def test_unpredicted_images_are_deferred(monkeypatch):
    """Populate must not seed a sentinel row before the detector has run.

    The predict cohort requires "no live head/tail label", so an image
    populated first would leave that cohort permanently and never be predicted
    — the same starvation the laser side hit on dive 84.
    """
    laser = [_laser(1), _laser(2)]
    images_by_id = {1: _image(1, "a"), 2: _image(2, "b")}

    # Only image 2 has been visited by the detector.
    fs = _make_fs_client(
        laser, [], images_by_id, predictions=[_StubPrediction(2)]
    )
    ls = _make_ls_client(returned_task_ids=[3001])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    n = await ActivityEnvironment().run(
        sut.populate_headtail_label_studio_project_activity, 42, 71
    )

    assert n == 1, "only the predicted image should be seeded"
    written = [c.args[1] for c in fs.labels.put_headtail_label.await_args_list]
    assert [row.image_id for row in written] == [2]
