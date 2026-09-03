"""Model representing a model-predicted laser dot for an image.

Written by the GPU laser-detector stage (fishsense-core `LaserDetector`) and
read by the laser populate step, which emits it as a Label Studio
pre-annotation so a labeler confirms/nudges rather than placing from scratch
(assisted review). Kept in its own table — separate from `LaserLabel` — so a
prediction never counts toward the human "valid laser" gates; a labeler's
confirmation still lands as a normal `LaserLabel` via the usual sync.
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import DateTime, Field, UniqueConstraint

from fishsense_api.models.model_base import ModelBase


class LaserPrediction(ModelBase, table=True):
    """A model-predicted laser dot, in rectified-image pixels (the space
    labelers place `LaserLabel.x/y`). One row per image — re-prediction
    upserts on the natural key."""

    __table_args__ = (UniqueConstraint("image_id", name="uq_laser_prediction_image"),)

    id: int | None = Field(default=None, primary_key=True)
    # x/y are None when the detector found no laser; confidence is always set.
    x: float | None = Field(default=None)
    y: float | None = Field(default=None)
    confidence: float = Field(default=0.0)
    # Rectified frame dimensions the x/y are relative to — the laser populate
    # step needs them to convert pixels to Label Studio keypoint percentages.
    width: int | None = Field(default=None)
    height: int | None = Field(default=None)
    # Laser colour read off the dot's own pixels: "red" / "green", or NULL
    # when there was no dot to sample or the channels were too close to call.
    # Advisory per row — populate takes the dive-level majority, because
    # colour is a property of the rig for the whole dive (143 prod dives are
    # entirely red, 88 entirely green) and a single frame is ~98% reliable.
    color: str | None = Field(default=None)
    # Signed R-G at the dot in 8-bit levels, positive for red. Recorded, not
    # gated on: it is what makes a close call auditable after the fact, the
    # same role `LaserDepth.residual_m` plays for the depth stage.
    color_margin: float | None = Field(default=None)
    # The detector found a dot, but outside the expected-laser region, so x/y
    # were dropped. Distinct from an ordinary non-detection (the model found
    # nothing): without this the two are indistinguishable, and a mis-sized
    # region would read as a model that had stopped working.
    rejected_out_of_region: bool = Field(default=False)
    # Which version of the laser-detector stage produced this row. The
    # stage-level cohort selects on a *mismatch* with the current version, so
    # improving the detector makes re-prediction an ordinary drainable cohort
    # instead of a hand-run backfill. NULL means "predates versioning", which
    # reads as stale exactly once. See `fishsense_shared.laser_predictor`.
    predictor_version: int | None = Field(default=None, index=True)
    # Recorded, never gated on — the same role `LaserDepth.residual_m` plays.
    # These are what answer "why did this frame come out that way" later;
    # `predictor_version` is the one thing anything decides on.
    checkpoint: str | None = Field(default=None)
    core_version: str | None = Field(default=None)
    # --- auto-accept gate verdict --------------------------------------
    # Whether this prediction may skip human review, decided by the
    # data-worker's `laser_label_validation.auto_accept` gate and consumed by
    # the api-worker's laser populate step. It is NOT a property of the
    # prediction alone: the gate fits the dive's whole prediction set and
    # requires them to agree, so a frame is auto-acceptable only in the
    # context of the dive it came from.
    #
    # `False` is the safe default and is load-bearing: a prediction the gate
    # has never seen must never read as auto-acceptable. It also means a
    # re-prediction *clears* the verdict for free, because the persist
    # activity constructs the row without these fields and the upsert merges
    # the whole model — which is correct, since the old verdict was computed
    # from a dot this row no longer holds. The gate must re-run.
    auto_accept: bool = Field(default=False)
    # The gate's own word for what happened, so a frame that was NOT
    # auto-accepted still says why (off_line / along_line_outlier /
    # audit_sample / dive_ineligible / no_prediction). Without it, "not
    # auto-accepted" collapses a refusal to engage and a rejected dot into one
    # indistinguishable state, and the per-dive verdict mix is the primary
    # monitoring signal for this whole stage.
    gate_verdict: str | None = Field(default=None, index=True)
    # Perpendicular distance to the dive's fitted laser line, and position
    # along it as a robust z against the rest of the dive. Recorded, never
    # re-decided on — the role `LaserDepth.residual_m` plays. They are what
    # answer "how close was this call" after the fact, and they let the
    # thresholds be retuned against already-collected data without
    # re-predicting.
    line_offset_px: float | None = Field(default=None)
    line_position_z: float | None = Field(default=None)

    created_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)

    image_id: int | None = Field(default=None, foreign_key="image.id")
