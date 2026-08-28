"""Which version of the laser-detector stage produced a prediction.

The detectors get refined over time, and a prediction is only as good as the
stage that made it. Without a version on the row there is no way to ask "which
of these came from the old behaviour", so improving the stage leaves every
existing prediction stranded: `select_next_for_laser_prediction` keys on the
*absence* of a `LaserPrediction`, so one row — even a non-detection — removes
that image from the cohort permanently. Its own docstring called re-prediction
"a manual affair", and it was: the only recovery was deleting rows by hand.

This is the same idea `LaserDepth` and `Measurement` already use, where the
cohorts select on a *mismatch* against the calibration that produced them
rather than on absence. That is what makes a recompute an ordinary drainable
cohort instead of a hand-run backfill.

**Version the behaviour, not the checkpoint.** The obvious move is to key this
on the `.pt` file, and it would not work: the change that motivated it (adding
the expected-region gate and reading the laser's colour) altered what the stage
outputs for every frame while touching the checkpoint not at all. A
checkpoint-derived version would have reported "unchanged" and re-predicted
nothing. The checkpoint is one input to the behaviour; so are the region
polygon, the colour rule, the gate, and the fishsense-core `predict()` defaults.

**So bump this by hand, and bump it whenever the stage's output would differ
for an unchanged image.** `test_laser_predictor_version.py` pins it against the
inputs that are cheap to pin, but no test can catch every behaviour change —
this is a judgement call made at review time, which is exactly why it is a
literal in a diff rather than a hash computed at runtime. A hash would also
churn on changes that do not matter (a fishsense-core patch release) and, when
it did change, could not say why.

`checkpoint` and `core_version` are recorded alongside but never gated on, the
same way `LaserDepth.residual_m` is recorded and not gated: they are what let
you answer "why did this frame come out that way" months later, without
deciding anything on their own.

Rows written before this existed carry NULL, which reads as stale exactly once
and then drains — again matching the laser-depth rollout.
"""

from __future__ import annotations

__all__ = ["LASER_PREDICTOR_VERSION", "laser_model_version_tag"]

#: Bumped when the laser-detector stage's output would change for an unchanged
#: image. History:
#:   1 — the original stage: fishsense-core `LaserDetector`, no output gate,
#:       no colour, pre-annotation hardcoded to "Red Laser".
#:   2 — 2026-08-28: predictions outside `LASER_REGION_POLYGON` are rejected,
#:       and the laser's colour is read off the dot so populate can label a
#:       dive by its own majority instead of always red.
LASER_PREDICTOR_VERSION: int = 2


def laser_model_version_tag(version: int = LASER_PREDICTOR_VERSION) -> str:
    """The `model_version` stamped on a Label Studio pre-annotation.

    Label Studio keeps predictions per task keyed by this string and allows
    several per task, so it is both what a labeler sees attributed to the model
    and what makes re-attaching predictions idempotent — the backfill skips a
    task that already carries the current tag. It was the bare constant
    "laser-detector" until 2026-08-28, which meant a task could not be told
    apart from one seeded by an older stage.
    """
    return f"laser-detector-v{version}"
