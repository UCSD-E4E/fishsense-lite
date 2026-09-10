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


class TestProjectsAreConfiguredToShowPredictions:
    """Attaching a prediction is not enough to show one.

    Label Studio surfaces predictions to annotators only for the version named
    in the *project's* `model_version`. Unset, a project with
    `show_collab_predictions=True` and hundreds of stored predictions still
    renders blank tasks -- which is how a labeler worked five frames of dive 94
    by hand with 334 invisible predictions sitting there.
    """

    class _LS:
        def __init__(self, versions):
            self.projects = self  # the client exposes `.projects.get/update`
            self._versions = dict(versions)
            self.updates = []
            self.raises_on = set()

        def get(self, id):  # noqa: A002  pylint: disable=redefined-builtin
            if id in self.raises_on:
                raise RuntimeError("boom")
            return type("_P", (), {"model_version": self._versions.get(id, "")})()

        def update(self, id, model_version):  # noqa: A002  pylint: disable=redefined-builtin
            self.updates.append((id, model_version))
            self._versions[id] = model_version

    def _run(self, ls, project_ids):
        import asyncio

        from fishsense_api_workflow_worker.activities.backfill_headtail_predictions_activity import (  # noqa: E501  pylint: disable=line-too-long
            _ensure_projects_show_predictions,
        )

        return asyncio.run(_ensure_projects_show_predictions(ls, project_ids))

    def test_an_unset_project_is_pointed_at_the_current_tier(self):
        from fishsense_shared.headtail_predictor import headtail_model_version_tag

        ls = self._LS({7: ""})
        assert self._run(ls, {7}) == 1
        assert ls.updates == [(7, headtail_model_version_tag())]

    def test_an_already_correct_project_is_left_alone(self):
        from fishsense_shared.headtail_predictor import headtail_model_version_tag

        ls = self._LS({7: headtail_model_version_tag()})
        assert self._run(ls, {7}) == 0
        assert not ls.updates

    def test_the_fallback_tier_is_upgraded_to_the_current_one(self):
        """A project can display exactly one version, and head/tail predictions
        are two-tier. Show SAM 3.1; fallback rows are queued for upgrade."""
        from fishsense_shared.headtail_predictor import (
            HEADTAIL_FALLBACK_PREDICTOR_VERSION,
            headtail_model_version_tag,
        )

        fallback = headtail_model_version_tag(HEADTAIL_FALLBACK_PREDICTOR_VERSION)
        ls = self._LS({7: fallback})
        assert self._run(ls, {7}) == 1
        assert ls.updates == [(7, headtail_model_version_tag())]

    def test_a_failure_does_not_break_the_backfill(self):
        """The predictions are attached and correct either way; display config
        is not worth failing an activity that already did its work."""
        from fishsense_shared.headtail_predictor import headtail_model_version_tag

        ls = self._LS({7: "", 8: ""})
        ls.raises_on = {7}
        assert self._run(ls, {7, 8}) == 1
        assert ls.updates == [(8, headtail_model_version_tag())]
