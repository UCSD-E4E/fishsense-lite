"""Unit tests for populate_laser_label_studio_project_activity.

Both clients are mocked. The activity has two failure modes worth
guarding: (a) silently skipping already-completed images is the
correctness contract — the notebook does this and re-running the
workflow must be safe; (b) LS `import_tasks` returning a different
count than we sent must abort before writing mismatched LaserLabel
rows.
"""

from __future__ import annotations

import base64

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_sdk.models.laser_prediction import LaserPrediction
from fishsense_api_workflow_worker.activities import (
    populate_laser_label_studio_project_activity as sut,
    populate_utils as sut_utils,
)


def _prediction(image_id: int, *, x=100.0, y=200.0, width=4000, height=3000):
    return LaserPrediction(
        id=image_id,
        image_id=image_id,
        x=x,
        y=y,
        confidence=0.9,
        width=width,
        height=height,
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
    image_id: int, *, completed: bool, project_id: int | None = 73
) -> LaserLabel:
    return LaserLabel(
        id=None,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=project_id,
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


def test_select_unlabeled_excludes_images_with_any_completed_label():
    images = [_image(1, "a"), _image(2, "b"), _image(3, "c")]
    existing = [_label(1, completed=True), _label(2, completed=False)]

    result = sut._select_unlabeled_images(images, existing, {i.id for i in images})  # pylint: disable=protected-access

    # Image 1 has a completed label -> excluded.
    # Image 2's only label is incomplete -> included.
    # Image 3 has no label at all -> included.
    assert [img.id for img in result] == [2, 3]


def test_select_unlabeled_treats_null_project_sentinel_as_unlabeled():
    """Contract pin: populate's per-image filter is on `completed`, not
    on `label_studio_project_id`. So an image whose only existing
    LaserLabel is a NULL-project sentinel (legacy prod state, ~2000
    such rows as of 2026-05-03) MUST still get a fresh task pushed
    when populate runs against a real project — otherwise a freshly
    deployed canonical project couldn't seed labels for any of those
    images. The cohort selector and resolver intentionally diverge
    here: they treat sentinels as 'no work needed' (so preprocess
    doesn't redo JPEGs), but populate treats them as 'no completed
    label, push a fresh task in this real project'.
    """
    images = [_image(1, "a"), _image(2, "b")]
    existing = [
        # Image 1: only a sentinel.
        _label(1, completed=False, project_id=None),
        # Image 2: completed real-project row -> already labeled.
        _label(2, completed=True, project_id=43),
    ]

    result = sut._select_unlabeled_images(images, existing, {i.id for i in images})  # pylint: disable=protected-access

    # Image 1 still needs a task; image 2 doesn't.
    assert [img.id for img in result] == [1]


def test_select_unlabeled_handles_multi_row_state():
    """Mirrors the prod state on dive 393: each image carries a
    completed row in project 43 plus an incomplete sentinel row in
    project NULL. The dict-collapse filter would resolve to either
    row depending on iteration order — the set-based filter doesn't.
    """
    images = [_image(1, "a"), _image(2, "b")]
    existing = [
        # Image 1: completed in 43 + incomplete sentinel.
        _label(1, completed=True, project_id=43),
        _label(1, completed=False, project_id=None),
        # Image 2: only an incomplete row.
        _label(2, completed=False, project_id=43),
    ]

    result = sut._select_unlabeled_images(images, existing, {i.id for i in images})  # pylint: disable=protected-access

    assert [img.id for img in result] == [2]


def test_build_task_uses_configured_url_base_and_dual_keys(monkeypatch):
    """Pinned: the LS task `data` must carry BOTH `image` and `img`
    keys with identical URLs. Legacy prod LS projects' labeling-config
    XML uses different conventions across stages and across project
    generations — emitting only one key gets `import_tasks` rejected
    with HTTP 400 ('img key is expected in task data') against the
    older projects. Reverting either key would re-introduce the
    populate regression observed on 2026-05-03.
    """
    monkeypatch.setenv("E4EFS_OBJECT_STORE__BUCKET", "fishsense-test")
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    # LS presigns this `s3://` URI via the per-project Garage source
    # storage. The prefix is the physical Garage key prefix the
    # data-worker wrote to (preprocess_jpeg), with the `.JPG` suffix.
    expected_url = "s3://fishsense-test/preprocess_jpeg/abc123.JPG"
    task = sut._build_task(_image(7, "abc123"))  # pylint: disable=protected-access

    assert task == {
        "data": {"image": expected_url, "img": expected_url},
        "annotations": [],
        "predictions": [],
    }


def test_select_unlabeled_gates_on_prediction_present():
    """An image with no LaserPrediction is deferred — populating it would
    stamp a LaserLabel and starve the predict cohort before the detector ran."""
    images = [_image(1, "a"), _image(2, "b")]
    # Only image 1 has been predicted.
    result = sut._select_unlabeled_images(images, [], {1})  # pylint: disable=protected-access
    assert [img.id for img in result] == [1]


def test_prediction_annotations_converts_pixels_to_percent():
    pred = _prediction(1, x=2000.0, y=1500.0, width=4000, height=3000)
    result = sut._prediction_annotations(pred)  # pylint: disable=protected-access
    assert len(result) == 1
    kp = result[0]["result"][0]
    assert kp["from_name"] == "laser" and kp["to_name"] == "img"
    assert kp["type"] == "keypointlabels"
    assert kp["original_width"] == 4000 and kp["original_height"] == 3000
    assert kp["value"]["x"] == 50.0  # 2000/4000*100
    assert kp["value"]["y"] == 50.0  # 1500/3000*100
    assert kp["value"]["keypointlabels"] == ["Red Laser"]


def test_prediction_annotations_empty_for_none_or_missing_dims():
    # pylint: disable=protected-access
    assert not sut._prediction_annotations(None)
    assert not sut._prediction_annotations(_prediction(1, x=None, y=None))
    assert not sut._prediction_annotations(
        LaserPrediction(image_id=1, x=1.0, y=2.0, confidence=0.9, width=None, height=None)
    )


def _make_fs_client(
    images: List[Image],
    existing_labels: List[LaserLabel],
    predictions: List[LaserPrediction] | None = None,
):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    fs.images = MagicMock()
    fs.images.get = AsyncMock(return_value=images)

    fs.labels = MagicMock()
    fs.labels.get_laser_labels = AsyncMock(return_value=existing_labels)
    fs.labels.put_laser_label = AsyncMock()
    # Default: every image has a prediction, so the prediction-gate is a no-op
    # and the completed-label filter alone decides what gets populated. Tests
    # that exercise the gate pass an explicit subset.
    if predictions is None:
        predictions = [_prediction(image.id) for image in images]
    fs.labels.get_laser_predictions = AsyncMock(return_value=predictions)
    return fs


def _make_ls_client(returned_task_ids: List[int]):
    # Fake hosted LS: import creates tasks (assigning ids from
    # `returned_task_ids` in order) and `tasks.list` serves them back. The
    # import response carries NO task_ids -- hosted LS imports asynchronously.
    ls = MagicMock()
    ls.projects = MagicMock()
    _stored: List = []
    _ids = iter(returned_task_ids)

    def _import(project_id, request, return_task_ids=False):  # pylint: disable=unused-argument
        for task in request:
            _tid = next(_ids)
            _s3 = task["data"].get("image") or task["data"].get("img")
            _fileuri = base64.b64encode(_s3.encode()).decode()
            # hosted LS lists tasks with a per-task presign resolve-wrapper,
            # NOT the imported s3:// URL — mirror that so dedup is exercised.
            _stored.append(
                SimpleNamespace(
                    id=_tid,
                    data={"image": f"/tasks/{_tid}/resolve/?fileuri={_fileuri}"},
                )
            )
        return SimpleNamespace(import_=1)

    ls.projects.import_tasks = MagicMock(side_effect=_import)
    ls.tasks = MagicMock()
    ls.tasks.list = MagicMock(side_effect=lambda project=None: list(_stored))
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
async def test_rerun_does_not_reimport_existing_tasks(monkeypatch):
    """Hosted LS import is async and returns no task ids; the activity resolves
    ids by listing tasks and dedupes against ones already in the project. A
    second run (e.g. after a mid-activity failure) must NOT re-import — that is
    what stops the runaway task accumulation the old `imported.task_ids` crash
    caused."""
    images = [_image(1, "a"), _image(2, "b")]
    fs = _make_fs_client(images, existing_labels=[])
    ls = _make_ls_client(returned_task_ids=[1001, 1002])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    n1 = await ActivityEnvironment().run(
        sut.populate_laser_label_studio_project_activity, 42, 73
    )
    n2 = await ActivityEnvironment().run(
        sut.populate_laser_label_studio_project_activity, 42, 73
    )

    assert n1 == 2 and n2 == 2
    # Imported exactly once: the rerun found both tasks already present and
    # only re-resolved their ids to (re-)write the rows.
    assert ls.projects.import_tasks.call_count == 1
    written = [c.args[1] for c in fs.labels.put_laser_label.await_args_list]
    assert {label.label_studio_task_id for label in written} == {1001, 1002}


@pytest.mark.asyncio
async def test_publishes_project_after_import(monkeypatch):
    # Laser imports its whole selection in one pass, so the project is
    # complete after import -> publish it.
    images = [_image(1, "a"), _image(2, "b")]
    fs = _make_fs_client(images, [])
    ls = _make_ls_client(returned_task_ids=[1001, 1002])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.populate_laser_label_studio_project_activity, 42, 73
    )

    ls.projects.update.assert_called_once_with(id=73, is_published=True)


@pytest.mark.asyncio
async def test_does_not_publish_empty_project(monkeypatch):
    # Fully-labeled dive whose only rows point at an OLD project (99): the
    # fresh per-dive project (73) gets no tasks and must stay a hidden draft.
    images = [_image(1, "a")]
    existing = [_label(1, completed=True, project_id=99)]
    fs = _make_fs_client(images, existing)
    ls = _make_ls_client(returned_task_ids=[])

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    n = await ActivityEnvironment().run(
        sut.populate_laser_label_studio_project_activity, 42, 73
    )

    assert n == 0
    ls.projects.update.assert_not_called()
