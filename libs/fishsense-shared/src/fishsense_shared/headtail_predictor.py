"""Which version of the head/tail-predict stage produced a prediction.

Same idea, and the same justification, as `laser_predictor`: the cohort keys on
a *mismatch* with the current version rather than on the absence of a row, so
improving the stage drains as an ordinary cohort instead of needing rows
deleted by hand. Without it, one row — even an abstention — removes that image
from the cohort permanently.

This stage has a sharper need for it than the laser one did, because its output
already changed once before it shipped and both of the things that moved were
invisible to any checkpoint:

* the **mask backend** went from the fishsense-core Mask R-CNN to SAM3, which
  changed the output for every frame while touching no `.pt` file this repo
  owns;
* the **crop window** is a *tuned* parameter (1800x1350), chosen from a sweep
  against held-out frames. Re-tuning it changes every prediction and is exactly
  the kind of change a checkpoint hash would report as "unchanged".

**Version the behaviour, not the model.** Bump this by hand whenever the
stage's output would differ for an unchanged image — a different backend, a
different crop, a different prompt, a changed confidence band. It is a literal
in a diff rather than a hash computed at runtime so that the bump is a
judgement made at review time, and so it does not churn on a dependency patch
release that changes nothing.

`checkpoint` and `core_version` are recorded alongside but never gated on: they
answer "why did this frame come out that way" months later without deciding
anything on their own.

Rows written before this existed carry NULL, which reads as stale exactly once
and then drains.
"""

from __future__ import annotations

__all__ = [
    "HEADTAIL_PREDICTOR_VERSION",
    "HEADTAIL_CROP_HEIGHT",
    "HEADTAIL_CROP_WIDTH",
    "headtail_model_version_tag",
]

#: Bumped when the head/tail-predict stage's output would change for an
#: unchanged image. History:
#:   1 — 2026-09-03: initial stage. SAM3 concept prompt ["fish"] on an
#:       1800x1350 crop centred on the validated laser dot, keypointed by
#:       `fishsense_core.fish.FishHeadTailDetector`.
HEADTAIL_PREDICTOR_VERSION = 1

#: The laser-centred crop fed to the mask backend, in rectified-image pixels.
#:
#: Tuned, not chosen: a sweep over 1000/1400/1800/2200/3000-wide windows on 80
#: frames, then re-run on 70 held-out frames from 63 other dives. The tuning set
#: preferred 3000x2250, which came *last* on held-out — that difference is why
#: the held-out set exists. 1800x1350 was the only size good on both.
#:
#: It sits between two failure modes. Too small and the window cuts the fish:
#: at 1000x750 only 90% of frames contain both human keypoints, and fork error
#: is the worst measured. Too large and resolution is handed back: 3000x2250
#: finds the fewest fish. Treat 1400-2200 as a plateau rather than 1800 as an
#: optimum, and bump `HEADTAIL_PREDICTOR_VERSION` if it moves.
HEADTAIL_CROP_WIDTH = 1800
HEADTAIL_CROP_HEIGHT = 1350


def headtail_model_version_tag() -> str:
    """The Label Studio `model_version` stamped on every pre-annotation.

    **This is an idempotency key, not a log line.** The backfill activity keys
    on `(task_id, model_version)` to decide whether a task already carries this
    stage's prediction, so the tag must be a pure function of the stage's
    *behaviour* and of nothing else.

    It used to interpolate `checkpoint`, which is the pod-local filesystem path
    the checkpoint happened to be cached at — so moving the cache directory, or
    the PVC mounting at a different point, changed the key for output that was
    byte-identical and stacked a second prediction onto every already-seeded
    task. `core_version` had the same flaw one release removed.

    Both are still recorded on the `HeadTailPrediction` row, which is where
    "why did this frame come out that way" belongs; they simply do not belong
    in a key. `HEADTAIL_PREDICTOR_VERSION` is the thing that changes when the
    output changes, and it is now the only thing in here that can vary.
    """
    return (
        f"v{HEADTAIL_PREDICTOR_VERSION}"
        f" crop={HEADTAIL_CROP_WIDTH}x{HEADTAIL_CROP_HEIGHT}"
    )
