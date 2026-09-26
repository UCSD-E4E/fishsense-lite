"""`judge_dive` is the one definition of what the laser validator decides.

Two callers act on it and must never disagree: the hourly validator, which
supersedes what it flags, and the remediation tool, which revives superseded
labels the same judgement keeps. If they drifted, a revived label would be
superseded again within the hour — or worse, kept by the tool on grounds the
validator does not share.

The distinction remediation hangs on is *judged* versus *not flagged*. A dive
too small to fit, a line that is not confident, a reflection split and a
refused (>50%) fit all flag nothing — and none of them is a verdict that the
labels are good. Only a confident, un-refused fit judges.
"""

from __future__ import annotations

import numpy as np
from fishsense_api_sdk.models.laser_label import LaserLabel

from fishsense_data_processing_workflow_worker.laser_label_validation.judgement import (
    judge_dive,
)

SLOPE, INTERCEPT = 0.4, 100.0
NORMAL = np.array([-SLOPE, 1.0]) / np.hypot(SLOPE, 1.0)


def _label(label_id, image_id, xy, *, superseded=False, completed=True):
    return LaserLabel(
        id=label_id,
        label_studio_task_id=10_000 + label_id,
        label_studio_project_id=42,
        x=None if xy is None else float(xy[0]),
        y=None if xy is None else float(xy[1]),
        label="kp-1",
        updated_at=None,
        superseded=superseded,
        completed=completed,
        label_studio_json=None,
        image_id=image_id,
        user_id=None,
    )


def _dive(n=80, offsets=None, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(50.0, 1500.0, n)
    perp = rng.normal(0.0, 1.85, size=n)
    if offsets:
        for i, off in offsets.items():
            perp[i] += off
    pts = np.column_stack([xs, SLOPE * xs + INTERCEPT]) + perp[:, None] * NORMAL
    return [_label(i + 1, 1000 + i, p) for i, p in enumerate(pts)]


def test_a_clean_confident_dive_is_judged_and_flags_its_outliers():
    labels = _dive(offsets={5: 60.0, 40: -45.0})

    judgement = judge_dive(labels, calibration_image_ids=set())

    assert judgement.judged
    assert judgement.status == "flagged"
    assert judgement.flagged_ids == {labels[5].id, labels[40].id}


def test_a_clean_dive_with_nothing_to_flag_is_still_judged():
    judgement = judge_dive(_dive(), calibration_image_ids=set())

    assert judgement.judged
    assert judgement.status == "no_outliers"
    assert judgement.flagged_ids == set()


def test_too_few_positives_is_not_a_judgement():
    judgement = judge_dive(_dive(n=4), calibration_image_ids=set())

    assert not judgement.judged
    assert judgement.status == "too_few"


def test_a_line_that_is_not_confident_is_not_a_judgement():
    """Core flags nothing on a non-confident line. Reading that as "all kept"
    would revive every superseded label on the dive.

    Twelve dots on two distinct pixels: core 4.1.0 gives a line whose inliers
    sit on fewer than 3 distinct pixels a confidence of 0, deliberately."""
    dots = [(100.0, 100.0)] * 6 + [(200.0, 140.0)] * 6
    labels = [_label(i + 1, 1000 + i, p) for i, p in enumerate(dots)]

    judgement = judge_dive(labels, calibration_image_ids=set())

    assert not judgement.judged
    assert judgement.status == "not_confident"
    assert judgement.flagged_ids == set()


def test_a_reflection_split_is_not_a_judgement():
    labels = []
    for i in range(30):
        t = i * 30.0
        labels.append(_label(i + 1, 100 + i, (1945.0 + 0.3 * t, 800.0 + t)))
    for i in range(20):
        t = i * 45.0
        labels.append(_label(60 + i, 200 + i, (1900.0 + 0.3 * t, 800.0 + t)))

    judgement = judge_dive(labels, calibration_image_ids=set())

    assert not judgement.judged
    assert judgement.status == "reflection"
    assert judgement.reflection is not None


def test_the_fraction_gate_is_not_a_judgement(monkeypatch):
    from fishsense_data_processing_workflow_worker.laser_label_validation import (
        judgement as sut,
    )

    def flags_sixty_percent(xy, _fit, **_kwargs):
        flags = np.zeros(xy.shape[0], dtype=bool)
        flags[: int(0.6 * xy.shape[0])] = True
        return flags

    monkeypatch.setattr(sut, "flag_outliers", flags_sixty_percent)

    judgement = judge_dive(_dive(), calibration_image_ids=set())

    assert not judgement.judged
    assert judgement.status == "gate"
    assert judgement.flagged_ids == set()


def test_superseded_labels_are_judged_too():
    """The whole population, every run — a superseded outlier is still
    flagged, a superseded inlier is still kept."""
    labels = _dive(offsets={5: 60.0})
    labels[5].superseded = True
    labels[9].superseded = True

    judgement = judge_dive(labels, calibration_image_ids=set())

    assert labels[5].id in judgement.flagged_ids
    assert labels[9].id not in judgement.flagged_ids
    assert labels[9].id in judgement.kept_ids


def test_the_verdict_does_not_depend_on_input_order():
    labels = _dive(offsets={5: 60.0, 40: -45.0})
    shuffled = list(reversed(labels))

    assert (
        judge_dive(shuffled, calibration_image_ids=set()).flagged_ids
        == judge_dive(labels, calibration_image_ids=set()).flagged_ids
    )


def test_calibration_frames_get_the_coarse_rule_and_are_marked():
    labels = _dive()
    on_slate = labels[30]
    on_slate.y += 8.0  # a genuine slate dot a few px off the fish line

    judgement = judge_dive(labels, calibration_image_ids={on_slate.image_id})

    assert on_slate.id not in judgement.flagged_ids
    assert judgement.is_calibration(on_slate.id)
    assert not judgement.is_calibration(labels[0].id)


def test_labels_without_a_dot_are_not_part_of_the_population():
    labels = _dive() + [_label(900, 9000, None)]

    judgement = judge_dive(labels, calibration_image_ids=set())

    assert 900 not in judgement.kept_ids | judgement.flagged_ids
