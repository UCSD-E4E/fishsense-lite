"""Unit tests for the lattice-verification task and prediction builders.

The overlay itself is tested on the data-worker side. What is pinned here is
the translation from a `CheckerboardLatticeRender` into a Label Studio task:
the coordinate conversion, which frames become tasks at all, and the control
names, which are a silent contract with the labeling config.
"""

from __future__ import annotations

import pytest
from fishsense_shared import CheckerboardLatticeRender

from fishsense_api_workflow_worker.activities import (
    populate_checkerboard_lattice_label_studio_project_activity as sut,
)
from fishsense_api_workflow_worker.activities.populate_utils import ImportResult


def _render(**overrides) -> CheckerboardLatticeRender:
    kwargs = {
        "image_id": 7,
        "checksum": "a" * 32,
        "detected_rows": 2,
        "detected_cols": 3,
        "median_spacing_px": 32.0,
        "corners": [
            [0.0, 0.0],
            [200.0, 0.0],
            [400.0, 0.0],
            [0.0, 150.0],
            [200.0, 150.0],
            [400.0, 150.0],
        ],
        "width": 400,
        "height": 300,
    }
    kwargs.update(overrides)
    return CheckerboardLatticeRender(**kwargs)


def test_one_keypoint_per_detected_corner():
    """The labeler counts marks, so the count must be the detection's."""
    [prediction] = sut.lattice_predictions(_render())

    assert len(prediction["result"]) == 6


def test_keypoints_are_percentages_of_the_rectified_frame():
    """Label Studio stores keypoints as percentages, not pixels.

    Getting this wrong does not error — it silently scatters the marks, and a
    labeler would report a lattice fault that is really a units bug, which is
    the one failure mode that would poison this study's answer.
    """
    [prediction] = sut.lattice_predictions(_render())
    values = [item["value"] for item in prediction["result"]]

    assert (values[0]["x"], values[0]["y"]) == (0.0, 0.0)
    # (200, 150) of a 400x300 frame is dead centre.
    assert (values[4]["x"], values[4]["y"]) == (50.0, 50.0)
    # (400, 0) is the right edge, top.
    assert (values[2]["x"], values[2]["y"]) == (100.0, 0.0)


def test_results_carry_the_frame_dimensions():
    [prediction] = sut.lattice_predictions(_render())

    assert {i["original_width"] for i in prediction["result"]} == {400}
    assert {i["original_height"] for i in prediction["result"]} == {300}


def test_control_names_match_the_labeling_config():
    """`from_name`/`to_name` are a silent contract with the project XML.

    A mismatch stores the prediction and renders nothing — the same class of
    invisible-prediction failure that cost five frames of dive 94 by hand.
    """
    from fishsense_api_workflow_worker.activities import (
        create_checkerboard_lattice_label_studio_project_activity as create_module,
    )

    xml = create_module.CHECKERBOARD_LATTICE_LABELING_CONFIG_XML

    [prediction] = sut.lattice_predictions(_render())
    item = prediction["result"][0]

    assert f'name="{item["from_name"]}"' in xml
    assert f'<Image name="{item["to_name"]}"' in xml
    assert item["type"] == "keypointlabels"
    assert f'value="{item["value"]["keypointlabels"][0]}"' in xml


def test_predictions_carry_a_model_version():
    """Without one the project's `model_version` cannot select them.

    Inline-at-import is what makes them visible at all: `import_tasks` sets the
    project field for free only when the task carries its predictions. This
    stage must never backfill predictions onto pre-existing tasks.
    """
    [prediction] = sut.lattice_predictions(_render())

    assert prediction["model_version"] == sut.LATTICE_MODEL_VERSION


@pytest.mark.parametrize(
    "overrides",
    [
        {"corners": None, "detected_rows": None, "detected_cols": None},
        {"width": None},
        {"height": None},
        {"corners": []},
    ],
)
def test_nothing_placeable_yields_no_prediction(overrides):
    assert not sut.lattice_predictions(_render(**overrides))


def test_a_skipped_frame_becomes_no_task():
    """The study is about lattices that actually fed a fit.

    A frame the detector rejected is not evidence either way about the
    lattice, and putting it in front of a labeler only dilutes the queue with
    frames whose honest answer is "there is nothing to judge here".
    """
    skipped = CheckerboardLatticeRender(
        image_id=9, checksum="b" * 32, skip_reason="no_usable_board"
    )

    assert sut.renders_worth_labeling([_render(), skipped]) == [_render()]


def test_a_rendered_frame_without_corners_becomes_no_task():
    """Belt and braces on the data-worker contract.

    A render with no `skip_reason` but no corners would otherwise produce a
    task with an image and no marks, which a labeler can only read as a
    detection failure — a verdict about the wrong thing.
    """
    hollow = _render(corners=None, detected_rows=None, detected_cols=None)

    assert sut.renders_worth_labeling([hollow]) == []


def test_task_data_points_at_the_lattice_jpeg_prefix():
    """Not the stage-0.1 prefix: same checksum, different picture.

    Pointing at `preprocess_jpeg` would serve the laser project's frame here
    and, worse, imply this stage could write there — overwriting what laser
    labelers are looking at.
    """
    from fishsense_shared.object_store import CHECKERBOARD_LATTICE_JPEG_FOLDER

    assert sut.LATTICE_FOLDER == CHECKERBOARD_LATTICE_JPEG_FOLDER
    assert CHECKERBOARD_LATTICE_JPEG_FOLDER != "preprocess_jpeg"


def test_task_data_leaks_nothing_about_the_detection():
    """Every data key is a sortable column in the LS Data Manager.

    `median_spacing_px` is the diagnostic itself — a coarse lattice runs at
    roughly twice the spacing, so one sort groups the faults before anyone has
    looked at a frame. The grid shape leaks it more weakly, and `image_id` is
    monotonic within a dive, so sorting on it reassembles the per-dive blocks
    the shuffle exists to break up. All three unblind the study from inside the
    labeling UI, where nobody would think to look for them.
    """
    data = sut._build_task(_render())["data"]  # pylint: disable=protected-access

    assert set(data) == {"image", "img"}


def test_a_verdict_is_still_traceable_to_its_frame():
    """Blinding must not cost traceability, or the verdicts are unusable.

    The checksum rides in the image URL — the same identifier every populate
    path here keys on — and checksums sort randomly, so it groups nothing.
    """
    render = _render()
    data = sut._build_task(render)["data"]  # pylint: disable=protected-access

    assert render.checksum in data["image"]


# --- the activity's handling of ImportResult -------------------------------
#
# Both properties below were bugs found in review on 2026-09-11, and both fail
# silently: one returns a wrong-typed value that nothing validates, the other
# publishes a project that merely looks finished.


def _patch_import(monkeypatch, result, published: list):
    async def _fake_import(**_kwargs):
        return result

    async def _fake_publish(project_id: int):
        published.append(project_id)

    monkeypatch.setattr(sut, "import_tasks_and_record_labels", _fake_import)
    monkeypatch.setattr(sut, "publish_label_studio_project", _fake_publish)


@pytest.mark.asyncio
async def test_the_activity_returns_a_count_not_the_result_tuple(monkeypatch):
    """`import_tasks_and_record_labels` returns `ImportResult`, not an int.

    Returning it from an activity annotated `-> int` raises nothing: Temporal
    serialises the NamedTuple as the two-element list `[recorded, deferred]`,
    the parent workflow sums it into its own `int` return, and the number a
    human reads as "tasks imported" is a list.
    """
    published: list[int] = []
    _patch_import(monkeypatch, ImportResult(recorded=3, deferred=0), published)

    imported = await sut.populate_checkerboard_lattice_label_studio_project_activity(
        7, [_render()]
    )

    assert imported == 3
    assert isinstance(imported, int)


@pytest.mark.asyncio
async def test_a_complete_import_publishes_the_project(monkeypatch):
    published: list[int] = []
    _patch_import(monkeypatch, ImportResult(recorded=2, deferred=0), published)

    await sut.populate_checkerboard_lattice_label_studio_project_activity(
        7, [_render()]
    )

    assert published == [7]


@pytest.mark.asyncio
async def test_a_deferred_import_leaves_the_project_unpublished(monkeypatch):
    """Hosted LS imports asynchronously, so a deferred batch means tasks exist
    that are not yet listable.

    Publishing then hands annotators a half-populated task list. The scheduled
    stages survive that because their next firing reconciles it; this study is
    on-demand and has no next firing, so the partial project is simply what
    gets labeled.
    """
    published: list[int] = []
    _patch_import(monkeypatch, ImportResult(recorded=2, deferred=5), published)

    imported = await sut.populate_checkerboard_lattice_label_studio_project_activity(
        7, [_render()]
    )

    assert imported == 2
    assert published == []


@pytest.mark.asyncio
async def test_nothing_worth_labeling_imports_nothing(monkeypatch):
    published: list[int] = []
    _patch_import(monkeypatch, ImportResult(recorded=99, deferred=0), published)

    skipped = CheckerboardLatticeRender(
        image_id=9, checksum="b" * 32, skip_reason="no_usable_board"
    )
    imported = await sut.populate_checkerboard_lattice_label_studio_project_activity(
        7, [skipped]
    )

    assert imported == 0
    assert published == []
