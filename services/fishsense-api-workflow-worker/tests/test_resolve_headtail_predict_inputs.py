"""Selection tests for the head/tail predict resolver.

The resolver must mirror the API cohort selector exactly. CLAUDE.md is blunt
about why: if the dispatched per-image work does not match what the cohort
promised, the dive can never drain and the parent re-fires on it every hour
forever.
"""

from __future__ import annotations


from fishsense_api_workflow_worker.activities.resolve_headtail_predict_inputs_activity import (  # noqa: E501  pylint: disable=line-too-long
    select_images_needing_prediction,
)
from fishsense_shared.headtail_predictor import HEADTAIL_PREDICTOR_VERSION


class _Image:
    def __init__(self, image_id, checksum="abc", is_canonical=True):
        self.id = image_id
        self.checksum = checksum
        self.is_canonical = is_canonical


class _Laser:
    def __init__(self, label_id, image_id, x=100.0, y=200.0, completed=True):
        self.id = label_id
        self.image_id = image_id
        self.x = x
        self.y = y
        self.completed = completed


class _HeadTail:
    def __init__(self, image_id, completed=True):
        self.image_id = image_id
        self.completed = completed


class _Prediction:
    def __init__(self, image_id, version=None, laser_label_id=None):
        self.image_id = image_id
        self.predictor_version = (
            HEADTAIL_PREDICTOR_VERSION if version is None else version
        )
        self.laser_label_id = laser_label_id


def _run(images, lasers, labels, predictions):
    return select_images_needing_prediction(images, lasers, labels, predictions)


def test_selects_an_unpredicted_image_with_a_valid_laser():
    picked = _run([_Image(1)], [_Laser(101, 1)], [], [])
    assert [p.image_id for p in picked] == [1]
    assert picked[0].laser_points == [[100.0, 200.0]]
    assert picked[0].laser_label_ids == [101]


def test_skips_an_image_with_no_laser():
    """The dot is the crop centre; without one there is nothing to predict on."""
    assert not _run([_Image(1)], [], [], [])


def test_skips_a_laser_missing_coordinates():
    assert not _run([_Image(1)], [_Laser(101, 1, x=None)], [], [])


def test_skips_an_incomplete_laser():
    assert not _run([_Image(1)], [_Laser(101, 1, completed=False)], [], [])


def test_skips_an_image_a_human_already_labelled():
    assert not _run([_Image(1)], [_Laser(101, 1)], [_HeadTail(1)], [])


def test_an_incomplete_headtail_row_is_not_a_label():
    """Populate seeds sentinel rows; those must not starve the detector — the
    dive-84 case on the laser side."""
    picked = _run([_Image(1)], [_Laser(101, 1)], [_HeadTail(1, completed=False)], [])
    assert [p.image_id for p in picked] == [1]


def test_skips_an_image_with_a_current_prediction():
    assert not _run([_Image(1)], [_Laser(101, 1)], [], [_Prediction(1)])


def test_reselects_a_stale_version():
    picked = _run([_Image(1)], [_Laser(101, 1)], [], [_Prediction(1, version=0)])
    assert [p.image_id for p in picked] == [1]


def test_reselects_a_null_version():
    """Pre-versioning rows carry NULL, and those are exactly what a bump
    exists to revisit."""
    picked = _run([_Image(1)], [_Laser(101, 1)], [], [_Prediction(1, version=None)])
    picked = _run(
        [_Image(1)], [_Laser(101, 1)], [], [_Prediction(1, version="unset")]
    )
    assert [p.image_id for p in picked] == [1]


def test_reselects_when_the_predictions_laser_was_superseded():
    """`get_laser_labels` filters superseded server-side, so a prediction
    naming a label that is no longer in the live set was made from a dot since
    dead-lettered — the mask may be of the wrong thing entirely."""
    picked = _run(
        [_Image(1)],
        [_Laser(102, 1)],  # 101 is gone: superseded
        [],
        [_Prediction(1, laser_label_id=101)],
    )
    assert [p.image_id for p in picked] == [1]
    assert picked[0].laser_label_ids == [102], "must use the surviving dot"


def test_skips_non_canonical_images():
    assert not _run([_Image(1, is_canonical=False)], [_Laser(101, 1)], [], [])


def test_carries_every_valid_dot_in_order():
    picked = _run(
        [_Image(1)],
        [_Laser(101, 1, x=10.0, y=20.0), _Laser(102, 1, x=11.0, y=21.0)],
        [],
        [],
    )
    assert picked[0].laser_points == [[10.0, 20.0], [11.0, 21.0]]
    assert picked[0].laser_label_ids == [101, 102]


class TestJpegPresenceGate:  # pylint: disable=protected-access
    """The predict stage reads the stage-5.1 JPEG, so it must not be
    dispatched for an image stage 5.1 has not rendered yet.

    +30 renders and +32 predicts, and stage 5.1 is a rawpy pass over a whole
    dive, so the two overlap routinely. Without the gate the child retries a
    `NoSuchKey` with no ceiling until its own execution timeout expires.
    """

    @staticmethod
    async def _gate(monkeypatch, present_checksums):
        from fishsense_api_workflow_worker.activities import (
            resolve_headtail_predict_inputs_activity as mod,
        )

        class _Store:
            async def has_processed_jpeg(self, _folder, checksum):
                return checksum in present_checksums

        monkeypatch.setattr(mod, "open_object_store_client", _Store)
        monkeypatch.setattr(mod.activity, "heartbeat", lambda *a, **k: None)
        return mod

    async def test_defers_images_without_a_rendered_jpeg(self, monkeypatch):
        from fishsense_shared.preprocess_contracts import PredictHeadtailImage

        mod = await self._gate(monkeypatch, {"aaa"})
        candidates = [
            PredictHeadtailImage(
                image_id=1, checksum="aaa", laser_points=[[1.0, 2.0]], laser_label_ids=[1]
            ),
            PredictHeadtailImage(
                image_id=2, checksum="bbb", laser_points=[[3.0, 4.0]], laser_label_ids=[2]
            ),
        ]
        kept = await mod._only_with_rendered_jpeg(candidates)
        assert [c.image_id for c in kept] == [1]

    async def test_empty_input_touches_no_object_store(self, monkeypatch):
        from fishsense_api_workflow_worker.activities import (
            resolve_headtail_predict_inputs_activity as mod,
        )

        def _explode():
            raise AssertionError("must not open the object store for an empty dive")

        monkeypatch.setattr(mod, "open_object_store_client", _explode)
        assert not await mod._only_with_rendered_jpeg([])
