"""The remediation plan: which superseded laser labels to revive.

Before 2026-09-26 the hourly validator re-fitted only each dive's survivors and
eroded dives a run at a time; ~11,800 of prod's 14,523 superseded positives are
kept by one full-population judgement. The plan proposes reviving exactly
those — and nothing the validator would supersede again — under constraints
that are each a way a revival could do harm:

* only where the dive was actually JUDGED. "Not flagged" on a dive too small
  to fit, with an unconfident line, a reflection split or a refused (>50%) fit
  is an abstention; reading it as "keep" would revive everything there;
* never a label or dive on the operator's exclusion list (deliberate manual
  supersessions: dive 77's reflection recipe, 347's wild slate dots);
* only completed labels (an incomplete row was never a valid laser);
* reflection suspects are reported for review, never revived.

The report also says where a revival will need a human downstream: on a
calibration frame (existing extrinsics are never refit), and on an image that
already has another live label (stages 13/14 read an unordered "first" row).
"""

from __future__ import annotations

import numpy as np
from fishsense_api_sdk.models.laser_label import LaserLabel

from fishsense_data_processing_workflow_worker.laser_label_validation.remediation import (
    plan_dive,
    plan_digest,
)

SLOPE, INTERCEPT = 0.4, 100.0
NORMAL = np.array([-SLOPE, 1.0]) / np.hypot(SLOPE, 1.0)


def _label(label_id, image_id, xy, *, superseded=False, completed=True):
    return LaserLabel(
        id=label_id,
        label_studio_task_id=10_000 + label_id,
        label_studio_project_id=42,
        x=float(xy[0]),
        y=float(xy[1]),
        label="kp-1",
        updated_at=None,
        superseded=superseded,
        completed=completed,
        label_studio_json=None,
        image_id=image_id,
        user_id=None,
    )


def _dive(n=80, outliers=None, superseded=(), seed=0):
    """A prod-like dive; `superseded` indexes are marked superseded."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(50.0, 1500.0, n)
    perp = rng.normal(0.0, 1.85, size=n)
    for i, off in (outliers or {}).items():
        perp[i] += off
    pts = np.column_stack([xs, SLOPE * xs + INTERCEPT]) + perp[:, None] * NORMAL
    return [
        _label(i + 1, 1000 + i, p, superseded=i in superseded)
        for i, p in enumerate(pts)
    ]


def test_revives_eroded_labels_the_fit_keeps_and_not_the_ones_it_flags():
    labels = _dive(outliers={5: 60.0}, superseded={5, 9, 20})

    plan = plan_dive(7, labels, calibration_image_ids=set())

    assert plan.status == "flagged"
    assert plan.revive_ids == [labels[9].id, labels[20].id]
    assert plan.superseded_now == 3
    assert plan.superseded_after == 1


def test_an_unjudged_dive_revives_nothing():
    """Twelve dots on two pixels: core calls that line not confident, flags
    nothing — and nothing here says those superseded labels are good."""
    dots = [(100.0, 100.0)] * 6 + [(200.0, 140.0)] * 6
    labels = [
        _label(i + 1, 1000 + i, p, superseded=i < 4) for i, p in enumerate(dots)
    ]

    plan = plan_dive(7, labels, calibration_image_ids=set())

    assert plan.status == "not_confident"
    assert plan.revive_ids == []
    assert plan.superseded_after == plan.superseded_now == 4
    assert plan.unjudged_superseded == 4


def test_a_reflection_suspect_is_reported_and_never_revived():
    labels = []
    for i in range(30):
        t = i * 30.0
        labels.append(_label(i + 1, 100 + i, (1945.0 + 0.3 * t, 800.0 + t)))
    for i in range(20):
        t = i * 45.0
        labels.append(
            _label(60 + i, 200 + i, (1900.0 + 0.3 * t, 800.0 + t), superseded=True)
        )

    plan = plan_dive(77, labels, calibration_image_ids=set())

    assert plan.status == "reflection"
    assert plan.revive_ids == []
    assert plan.reflection_suspect is not None
    assert plan.reflection_suspect["n_secondary"] > 0


def test_excluded_labels_are_never_revived():
    labels = _dive(superseded={9, 20})

    plan = plan_dive(7, labels, set(), excluded_label_ids={labels[9].id})

    assert plan.revive_ids == [labels[20].id]
    assert plan.excluded_kept == [labels[9].id]


def test_an_excluded_dive_revives_nothing_and_says_why():
    labels = _dive(superseded={9, 20})

    plan = plan_dive(7, labels, set(), dive_excluded=True)

    assert plan.status == "excluded"
    assert plan.revive_ids == []
    assert plan.superseded_after == 2


def test_only_completed_labels_are_revived():
    labels = _dive(superseded={9, 20})
    labels[20].completed = False

    plan = plan_dive(7, labels, set())

    assert plan.revive_ids == [labels[9].id]


def test_revivals_needing_a_human_downstream_are_counted():
    labels = _dive(superseded={9, 20, 30})
    # 20 sits on a calibration (slate) frame; 30's image has a second, live label.
    second = _label(500, labels[30].image_id, (labels[30].x, labels[30].y))
    labels.append(second)

    plan = plan_dive(7, labels, calibration_image_ids={labels[20].image_id})

    assert set(plan.revive_ids) == {labels[9].id, labels[20].id, labels[30].id}
    assert plan.revive_on_calibration_frames == [labels[20].id]
    assert plan.revive_on_images_with_another_live_label == [labels[30].id]


def test_a_live_label_is_never_in_the_plan():
    plan = plan_dive(7, _dive(), set())

    assert plan.revive_ids == []
    assert plan.superseded_now == 0


def test_the_digest_names_exactly_the_revivals_and_ignores_order():
    a = plan_dive(7, _dive(superseded={9}), set())
    b = plan_dive(8, _dive(superseded={20}, seed=1), set())

    assert plan_digest([a, b]) == plan_digest([b, a])
    assert plan_digest([a]) != plan_digest([a, b])


def test_the_report_row_is_json_ready():
    import json

    plan = plan_dive(7, _dive(outliers={5: 60.0}, superseded={5, 9}), set())

    row = json.loads(json.dumps(plan.to_dict()))
    assert row["dive_id"] == 7
    assert row["revive_ids"] == [10]
    assert row["positives"] == 80
