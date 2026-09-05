"""Unit tests for populate_laser_label_studio_project_activity.

Both clients are mocked. The activity has two failure modes worth
guarding: (a) silently skipping already-completed images is the
correctness contract — the notebook does this and re-running the
workflow must be safe; (b) LS `import_tasks` returning a different
count than we sent must abort before writing mismatched LaserLabel
rows.
"""

from __future__ import annotations


from typing import List
from unittest.mock import AsyncMock, MagicMock

from types import SimpleNamespace

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_sdk.models.laser_prediction import LaserPrediction
from fishsense_api_workflow_worker.activities import (

    populate_laser_label_studio_project_activity as sut,
    populate_utils as sut_utils,
)

# Shared with the other populate suites — see `worker_tests_support.populate`.
from worker_tests_support.populate import (  # noqa: F401
    image as _image,
    fake_label_studio_client as _make_ls_client,
)


@pytest.fixture(autouse=True)
def _default_store_all_present(monkeypatch):
    """Default: every image's laser JPEG is in Garage, so the JPEG-gate is a
    no-op and the existing tests decide populate purely on the label/prediction
    state. Tests that exercise the gate override open_object_store_client."""
    store = MagicMock()
    store.has_processed_jpeg = AsyncMock(return_value=True)
    monkeypatch.setattr(sut, "open_object_store_client", lambda: store)
    return store


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
    from fishsense_api_workflow_worker import config as cfg
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
async def test_defers_image_whose_laser_jpeg_is_not_in_garage(monkeypatch):
    """An image with a prediction + no completed label but NO laser JPEG in
    Garage must NOT be populated — its LS task would 404 (NoSuchKey).
    Regression for the project-276057 broken tasks on dive 60's pre-Garage
    stragglers."""
    images = [_image(1, "has-jpeg"), _image(2, "missing-jpeg")]
    fs = _make_fs_client(images, existing_labels=[])  # both predicted, none done
    ls = _make_ls_client(returned_task_ids=[2001])

    store = MagicMock()

    async def _has(_folder, checksum):
        return checksum != "missing-jpeg"

    store.has_processed_jpeg = AsyncMock(side_effect=_has)
    monkeypatch.setattr(sut, "open_object_store_client", lambda: store)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    n = await ActivityEnvironment().run(
        sut.populate_laser_label_studio_project_activity, 42, 73
    )

    assert n == 1  # only the image whose JPEG exists got a task
    _args, kwargs = ls.projects.import_tasks.call_args
    imported = [t["data"].get("image", "") for t in kwargs["request"]]
    assert len(imported) == 1
    assert any("missing-jpeg" not in u for u in imported)
    assert not any("missing-jpeg" in u for u in imported)
    assert fs.labels.put_laser_label.await_count == 1


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


# ------------------------- dive-level laser colour ---------------------------
#
# The pre-annotation hardcoded "Red Laser" until 2026-08-28, which is wrong for
# ~a quarter of prod (143 dives entirely red, 88 entirely green). Colour is a
# rig property for a whole dive: the 31 "mixed" dives carry a 1.2% minority,
# which is labeler slips rather than a laser changing colour mid-dive. So the
# per-frame reads are votes and the dive's majority is applied to every task.


def _pred(color):
    return SimpleNamespace(image_id=1, color=color)


def test_unanimous_green_dive_is_labelled_green():
    from fishsense_api_workflow_worker.activities.populate_laser_label_studio_project_activity import (  # pylint: disable=line-too-long
        dive_laser_label,
    )

    assert dive_laser_label([_pred("green")] * 5) == "Green Laser"


def test_a_single_misread_frame_does_not_split_the_dive():
    """The case the corpus actually showed: 10% of one dive's frames read the
    wrong colour. Every task in the dive must still get the same label -- a
    labeler seeing both colours inside one dive cannot tell which to trust."""
    from fishsense_api_workflow_worker.activities.populate_laser_label_studio_project_activity import (  # pylint: disable=line-too-long
        dive_laser_label,
    )

    votes = [_pred("red")] * 47 + [_pred("green")] * 6
    assert dive_laser_label(votes) == "Red Laser"


def test_abstentions_do_not_count_as_votes():
    """`color is None` means the classifier declined (no dot, or channels too
    close to call). Counting those as red would let a dive with two real green
    reads and fifty abstentions come out red."""
    from fishsense_api_workflow_worker.activities.populate_laser_label_studio_project_activity import (  # pylint: disable=line-too-long
        dive_laser_label,
    )

    assert dive_laser_label([_pred(None)] * 50 + [_pred("green")] * 2) == "Green Laser"


def test_no_votes_falls_back_to_the_more_common_colour():
    from fishsense_api_workflow_worker.activities.populate_laser_label_studio_project_activity import (  # pylint: disable=line-too-long
        dive_laser_label,
    )

    assert dive_laser_label([]) == "Red Laser"
    assert dive_laser_label([_pred(None), _pred("blue")]) == "Red Laser"


def test_a_tie_falls_back_rather_than_picking_arbitrarily():
    from fishsense_api_workflow_worker.activities.populate_laser_label_studio_project_activity import (  # pylint: disable=line-too-long
        dive_laser_label,
    )

    assert dive_laser_label([_pred("red"), _pred("green")]) == "Red Laser"


def test_predictions_without_a_colour_attribute_are_tolerated():
    """Rows written before the colour columns existed come back without one;
    they must not break populate for the dive."""
    from fishsense_api_workflow_worker.activities.populate_laser_label_studio_project_activity import (  # pylint: disable=line-too-long
        dive_laser_label,
    )

    legacy = SimpleNamespace(image_id=2)
    assert dive_laser_label([legacy, _pred("green"), _pred("green")]) == "Green Laser"


def test_the_chosen_label_reaches_the_keypoint_annotation():
    from fishsense_api_workflow_worker.activities.populate_laser_label_studio_project_activity import (  # pylint: disable=line-too-long
        _prediction_annotations,
    )

    prediction = SimpleNamespace(x=2000.0, y=1200.0, width=4014, height=3016)
    annotations = _prediction_annotations(prediction, "Green Laser")
    assert annotations[0]["result"][0]["value"]["keypointlabels"] == ["Green Laser"]


# --- auto-accepted frames -----------------------------------------------------
#
# A prediction the gate cleared is imported as a completed ANNOTATION rather
# than a prediction. Label Studio's labeling stream serves only un-annotated
# tasks, so the labeler never sees it — but the project stays a complete record
# of the dive and any frame can be reopened and corrected later, which a
# task-less direct write would not allow.
#
# Coordinates are deliberately NOT written here. The hourly sync remains the
# single writer of label x/y, reading them back out of Label Studio exactly as
# it does for a human annotation, so there is one code path and no way for
# populate and sync to disagree about the same frame.


def _accepted(image_id: int, **kwargs):
    prediction = _prediction(image_id, **kwargs)
    prediction.auto_accept = True
    prediction.gate_verdict = "auto_accepted"
    return prediction


def test_auto_accepted_prediction_becomes_an_annotation_not_a_prediction():
    # pylint: disable=protected-access
    task = sut._build_task(_image(7, "abc123"), _accepted(7), "Red Laser")
    assert not task["predictions"]
    assert len(task["annotations"]) == 1
    result = task["annotations"][0]["result"]
    assert result[0]["value"]["keypointlabels"] == ["Red Laser"]


def test_auto_accepted_annotation_matches_the_shape_of_an_accepted_prediction():
    """`origin: prediction` is what Label Studio itself stamps when a labeler
    opens a pre-annotated task and submits it unchanged. Writing the same thing
    is truthful — the model placed the point and nothing moved it — and keeps
    an auto-accepted frame indistinguishable in shape from the 93% of human
    reviews that produced exactly this."""
    # pylint: disable=protected-access
    task = sut._build_task(_image(7, "abc123"), _accepted(7), "Red Laser")
    result = task["annotations"][0]["result"][0]
    assert result["origin"] == "prediction"
    assert result["from_name"] == "laser"
    assert result["type"] == "keypointlabels"


def test_auto_accepted_annotation_names_the_service_account(monkeypatch):
    """**The import path must say who wrote it; the token does not.**

    `ls.annotations.create` stamps `completed_by` from the authenticated user,
    so swapping the worker's token to the service account fixed that path. An
    *imported* annotation is different: Label Studio attributes it to the
    project owner, not the API caller. Prod proved it — the worker was already
    running as the bot at 18:07 on 2026-09-04, and populate still seeded 61
    annotations at 18:17 under the human who owned the project, then 164 more
    at 19:17.

    So this path has to name the account explicitly, from config, because
    there is nothing in the request for LS to infer it from.
    """
    # pylint: disable=protected-access
    monkeypatch.setattr(sut.settings.label_studio, "bot_user_id", 215238, raising=False)

    task = sut._build_task(_image(7, "abc123"), _accepted(7), "Red Laser")

    assert task["annotations"][0]["completed_by"] == 215238


def test_auto_accepted_annotation_omits_completed_by_when_unconfigured(monkeypatch):
    """Unset means omit the key, never send None.

    `bot_user_id` has to reach the slot through the OpenBao render, and that is
    a three-step manual rotation nobody can be relied on to complete in one go.
    Sending `completed_by: null` on a miss would be a hard LS validation error
    that fails populate for the whole dive; omitting it degrades to the old
    behaviour, which is wrong attribution but not a broken pipeline.
    """
    # pylint: disable=protected-access
    monkeypatch.setattr(sut.settings.label_studio, "bot_user_id", None, raising=False)

    task = sut._build_task(_image(7, "abc123"), _accepted(7), "Red Laser")

    assert "completed_by" not in task["annotations"][0]


def test_auto_accepted_annotation_is_not_marked_ground_truth():
    """Shape-matching a human review stops at the *geometry*, not the claim.

    `origin: prediction` is truthful — the model placed the point and nothing
    moved it. `ground_truth` is a different assertion: Label Studio treats it
    as "this annotation is definitive", the reference other work is scored
    against. An auto-accepted frame skipped review entirely, and the gate ships
    an audit sample precisely because some of these are expected to be wrong.

    Must be explicit. Left unset, Label Studio stamped `ground_truth: true` on
    import — verified on prod dive 520, all 37 annotations.
    """
    # pylint: disable=protected-access
    task = sut._build_task(_image(7, "abc123"), _accepted(7), "Red Laser")

    assert task["annotations"][0]["ground_truth"] is False


def test_auto_accepted_keypoint_carries_the_same_percentages_as_a_prediction():
    """The annotation must land on the pixel the detector chose. Sharing the
    conversion with `_prediction_annotations` is what guarantees it."""
    # pylint: disable=protected-access
    seeded = sut._build_task(_image(7, "abc123"), _prediction(7), "Red Laser")
    accepted = sut._build_task(_image(7, "abc123"), _accepted(7), "Red Laser")
    assert (
        accepted["annotations"][0]["result"][0]["value"]
        == seeded["predictions"][0]["result"][0]["value"]
    )


def test_a_prediction_the_gate_rejected_is_still_seeded_for_review():
    """Not auto-accepted is the normal path, not an error: the frame goes to a
    labeler with the model's guess as a pre-annotation, exactly as before the
    gate existed."""
    # pylint: disable=protected-access
    prediction = _prediction(7)
    prediction.auto_accept = False
    prediction.gate_verdict = "off_line"
    task = sut._build_task(_image(7, "abc123"), prediction, "Red Laser")
    assert not task["annotations"]
    assert len(task["predictions"]) == 1


def test_an_ungated_prediction_is_seeded_not_auto_accepted():
    """`auto_accept` defaults False, so a prediction the gate has never judged
    behaves exactly as it did before this feature."""
    # pylint: disable=protected-access
    task = sut._build_task(_image(7, "abc123"), _prediction(7), "Red Laser")
    assert not task["annotations"]
    assert len(task["predictions"]) == 1
