"""Attaching head/tail predictions to Label Studio tasks that already exist.

Populate seeds a task's pre-annotation once, at import time, and dedupes by
URL — so for a dive whose tasks already exist, a new `HeadTailPrediction`
changes the database and nothing a labeler sees. On the corpus that is 3,147
still-unlabelled tasks across 19 dives, which would otherwise never receive a
keypoint.
"""

from __future__ import annotations

from collections import Counter

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

    The tier is chosen from what the project *has*, not from a constant. A
    dive predicted entirely on the CPU fallback holds no SAM 3.1 predictions,
    so pinning the current tag would point it at a version it does not have.
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

    def _run(self, ls, tags_by_project):
        import asyncio

        from fishsense_api_workflow_worker.activities.backfill_headtail_predictions_activity import (  # noqa: E501  pylint: disable=line-too-long
            _ensure_projects_show_predictions,
        )

        return asyncio.run(_ensure_projects_show_predictions(ls, tags_by_project))

    def test_an_unset_project_is_pointed_at_the_tier_it_holds(self):
        from fishsense_shared.headtail_predictor import headtail_model_version_tag

        current = headtail_model_version_tag()
        ls = self._LS({7: ""})
        assert self._run(ls, {7: Counter({current: 5})}) == 1
        assert ls.updates == [(7, current)]

    def test_an_already_correct_project_is_left_alone(self):
        from fishsense_shared.headtail_predictor import headtail_model_version_tag

        current = headtail_model_version_tag()
        ls = self._LS({7: current})
        assert self._run(ls, {7: Counter({current: 5})}) == 0
        assert not ls.updates

    def test_a_mixed_dive_shows_the_current_tier(self):
        """A project displays exactly one version. Show SAM 3.1; the fallback
        rows on the same dive are queued for upgrade anyway."""
        from fishsense_shared.headtail_predictor import (
            HEADTAIL_FALLBACK_PREDICTOR_VERSION,
            headtail_model_version_tag,
        )

        current = headtail_model_version_tag()
        fallback = headtail_model_version_tag(HEADTAIL_FALLBACK_PREDICTOR_VERSION)
        ls = self._LS({7: ""})
        # Fallback predictions outnumber SAM 3.1 ones; the current tier still wins.
        assert self._run(ls, {7: Counter({fallback: 40, current: 2})}) == 1
        assert ls.updates == [(7, current)]

    def test_an_all_fallback_dive_is_pinned_to_the_fallback_tier(self):
        """The regression this guards: pinning the current tag on a dive with
        no SAM 3.1 predictions points the project at a version it does not
        have, so every task stays blank."""
        from fishsense_shared.headtail_predictor import (
            HEADTAIL_FALLBACK_PREDICTOR_VERSION,
            headtail_model_version_tag,
        )

        fallback = headtail_model_version_tag(HEADTAIL_FALLBACK_PREDICTOR_VERSION)
        ls = self._LS({7: ""})
        assert self._run(ls, {7: Counter({fallback: 40})}) == 1
        assert ls.updates == [(7, fallback)]

    def test_a_working_fallback_value_is_not_overwritten(self):
        """LS sets the field itself when tasks are imported with predictions
        inline. An all-fallback project already showing its own tier must be
        left alone, not switched to one it has none of."""
        from fishsense_shared.headtail_predictor import (
            HEADTAIL_FALLBACK_PREDICTOR_VERSION,
            headtail_model_version_tag,
        )

        fallback = headtail_model_version_tag(HEADTAIL_FALLBACK_PREDICTOR_VERSION)
        ls = self._LS({7: fallback})
        assert self._run(ls, {7: Counter({fallback: 40})}) == 0
        assert not ls.updates

    def test_a_failure_does_not_break_the_backfill(self):
        """The predictions are attached and correct either way; display config
        is not worth failing an activity that already did its work."""
        from fishsense_shared.headtail_predictor import headtail_model_version_tag

        current = headtail_model_version_tag()
        ls = self._LS({7: "", 8: ""})
        ls.raises_on = {7}
        tags = {7: Counter({current: 1}), 8: Counter({current: 1})}
        assert self._run(ls, tags) == 1
        assert ls.updates == [(8, current)]
