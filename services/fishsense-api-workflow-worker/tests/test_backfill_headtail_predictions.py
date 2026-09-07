"""Attaching head/tail predictions to Label Studio tasks that already exist.

Populate seeds a task's pre-annotation once, at import time, and dedupes by
URL — so for a dive whose tasks already exist, a new `HeadTailPrediction`
changes the database and nothing a labeler sees. On the corpus that is 3,147
still-unlabelled tasks across 19 dives, which would otherwise never receive a
keypoint.
"""

from __future__ import annotations

from fishsense_api_workflow_worker.activities.backfill_headtail_predictions_activity import (  # noqa: E501  pylint: disable=line-too-long
    select_attach_targets,
)


class _Prediction:  # pylint: disable=too-many-instance-attributes
    def __init__(self, image_id, status="predicted", head=(10.0, 20.0)):
        self.image_id = image_id
        self.status = status
        self.head_x, self.head_y = head
        self.tail_x, self.tail_y = (110.0, 25.0)
        self.width, self.height = 4000, 3000
        self.silhouette_ratio = 0.25
        self.rejected_low_confidence = False
        self.checkpoint = None
        self.core_version = None


class _Label:
    def __init__(self, image_id, task_id=900, project_id=71, completed=False,
                 superseded=False):
        self.image_id = image_id
        self.label_studio_task_id = task_id
        self.label_studio_project_id = project_id
        self.completed = completed
        self.superseded = superseded


def test_selects_an_incomplete_task_with_a_placeable_prediction():
    targets = select_attach_targets([_Prediction(1)], [_Label(1, task_id=900)])
    assert targets == {1: (900, 71)}


def test_skips_a_completed_task():
    """A labeler has already placed those points; a fresh pre-annotation beside
    them is noise at best, and an invitation to second-guess finished work at
    worst."""
    assert not select_attach_targets([_Prediction(1)], [_Label(1, completed=True)])


def test_skips_a_superseded_row():
    assert not select_attach_targets([_Prediction(1)], [_Label(1, superseded=True)])


def test_skips_an_abstention():
    """Nothing placeable — attaching an empty prediction would be noise."""
    assert not select_attach_targets(
        [_Prediction(1, status="no_detections", head=(None, None))], [_Label(1)]
    )


def test_skips_a_task_with_no_ls_ids():
    assert not select_attach_targets([_Prediction(1)], [_Label(1, task_id=None)])


def test_skips_an_image_with_no_prediction():
    assert not select_attach_targets([], [_Label(1)])


def test_first_non_superseded_task_per_image_wins():
    targets = select_attach_targets(
        [_Prediction(1)],
        [_Label(1, task_id=900), _Label(1, task_id=901)],
    )
    assert targets == {1: (900, 71)}
