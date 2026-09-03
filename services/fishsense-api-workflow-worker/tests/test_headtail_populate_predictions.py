"""Head/tail populate: seeding model keypoints as Label Studio pre-annotations.

Two keypoints per task rather than one, and both must carry the labels the
project's XML declares (`Snout` / `Fork` on `kp-1`) — the sync activity filters
annotations on `from_name == "kp-1"`, so a mismatch there silently produces
tasks whose labels never come back.
"""

from __future__ import annotations

import pytest

from fishsense_api_workflow_worker.activities.populate_headtail_label_studio_project_activity import (  # noqa: E501
    prediction_annotations,
    select_predicted_image_ids,
)


class _Prediction:
    def __init__(
        self,
        image_id=1,
        head=(100.0, 200.0),
        tail=(300.0, 220.0),
        width=4000,
        height=3000,
        status="predicted",
        silhouette_ratio=0.25,
        rejected_low_confidence=False,
    ):
        self.image_id = image_id
        self.head_x, self.head_y = head
        self.tail_x, self.tail_y = tail
        self.width = width
        self.height = height
        self.status = status
        self.silhouette_ratio = silhouette_ratio
        self.rejected_low_confidence = rejected_low_confidence


def test_emits_two_keypoints_labelled_snout_and_fork():
    out = prediction_annotations(_Prediction())
    assert len(out) == 1
    results = out[0]["result"]
    assert len(results) == 2
    labels = [r["value"]["keypointlabels"][0] for r in results]
    assert labels == ["Snout", "Fork"]


def test_keypoints_use_the_projects_from_name():
    """`sync_headtail_labels_*` filters on `from_name == "kp-1"`; anything else
    is seeded but never read back."""
    results = prediction_annotations(_Prediction())[0]["result"]
    assert {r["from_name"] for r in results} == {"kp-1"}
    assert {r["type"] for r in results} == {"keypointlabels"}


def test_pixels_convert_to_percentages_using_the_recorded_dims():
    out = prediction_annotations(
        _Prediction(head=(1000.0, 600.0), tail=(3000.0, 1500.0), width=4000, height=3000)
    )
    snout, fork = out[0]["result"]
    assert snout["value"]["x"] == pytest.approx(25.0)
    assert snout["value"]["y"] == pytest.approx(20.0)
    assert fork["value"]["x"] == pytest.approx(75.0)
    assert fork["value"]["y"] == pytest.approx(50.0)


def test_no_annotation_for_an_abstention():
    assert prediction_annotations(_Prediction(status="no_detections", head=(None, None))) == []
    assert prediction_annotations(None) == []


def test_no_annotation_when_frame_dims_are_missing():
    """Without dims the pixel->percentage conversion is undefined; seeding a
    guess would put the point somewhere arbitrary on the labeler's screen."""
    assert prediction_annotations(_Prediction(width=None)) == []


def test_low_confidence_is_seeded_as_a_task_with_no_prediction():
    """Outside the silhouette band the task is still created — the image needs
    labelling either way — but nothing is placed on it."""
    assert prediction_annotations(_Prediction(rejected_low_confidence=True)) == []


def test_silhouette_band_rejects_a_non_fish_shape():
    """Applied at seed time, not predict time, so the band can be retuned from
    rows already collected without re-predicting anything."""
    assert prediction_annotations(_Prediction(silhouette_ratio=0.02)) == []
    assert prediction_annotations(_Prediction(silhouette_ratio=0.9)) == []
    assert prediction_annotations(_Prediction(silhouette_ratio=0.25)) != []


def test_a_missing_ratio_is_not_treated_as_out_of_band():
    """None means "not recorded", which must not silently suppress every
    prediction written before the column existed."""
    assert prediction_annotations(_Prediction(silhouette_ratio=None)) != []


def test_prediction_gate_only_admits_predicted_images():
    """Populate seeds sentinel `HeadTailLabel` rows, and the predict cohort
    requires "no live label" — so seeding an unpredicted image first would
    remove it from the cohort before the detector ever ran, starving it
    permanently. Same trap the laser side documents."""
    predictions = [_Prediction(image_id=1), _Prediction(image_id=2, status="no_detections")]
    assert select_predicted_image_ids(predictions) == {1, 2}


def test_prediction_gate_counts_abstentions_as_predicted():
    """An abstention *is* a prediction attempt: the image has been visited, so
    holding it back from populate forever would strand it instead."""
    assert select_predicted_image_ids([_Prediction(image_id=9, status="headtail_failed")]) == {9}
