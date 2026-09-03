"""Model representing model-predicted head/tail keypoints for an image.

Written by the GPU head/tail predict stage — a SAM3 mask of the fish the
validated laser dot sits on, keypointed by `fishsense_core`'s
`FishHeadTailDetector` — and read by the head/tail populate step, which emits
it as a Label Studio pre-annotation so a labeler confirms or nudges rather than
placing from scratch.

Kept in its own table, separate from `HeadTailLabel`, so a prediction never
counts toward the human "valid head/tail" gates; a labeler's confirmation still
lands as an ordinary `HeadTailLabel` via the usual sync. Same reason
`LaserPrediction` is separate from `LaserLabel`.
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import DateTime, Field, UniqueConstraint

from fishsense_api.models.model_base import ModelBase


class HeadTailPrediction(ModelBase, table=True):
    """Predicted snout/fork keypoints, in rectified-image pixels — the space
    labelers place `HeadTailLabel.head_x/y` and `tail_x/y`. One row per image;
    re-prediction upserts on the natural key."""

    __table_args__ = (
        UniqueConstraint("image_id", name="uq_headtail_prediction_image"),
    )

    id: int | None = Field(default=None, primary_key=True)

    # All four are None on an abstention; `status` says which kind.
    #
    # Stored already lifted out of the laser-centred crop the model ran on, so
    # they are directly comparable with a labeler's own clicks. `crop_x/crop_y`
    # below records the window they were lifted from.
    head_x: float | None = Field(default=None)
    head_y: float | None = Field(default=None)
    tail_x: float | None = Field(default=None)
    tail_y: float | None = Field(default=None)

    # Rectified frame dimensions the coordinates are relative to — the populate
    # step needs them to convert pixels to Label Studio keypoint percentages.
    width: int | None = Field(default=None)
    height: int | None = Field(default=None)

    # Which fish, and how fish-shaped it was. `silhouette_ratio` is
    # `mask_area_px / length**2`; a real fish silhouette runs ~0.15-0.30, and
    # the band is the confidence gate applied at seed time rather than here, so
    # it can be retuned from data already collected without re-predicting.
    mask_area_px: int | None = Field(default=None)
    silhouette_ratio: float | None = Field(default=None)

    # Origin of the laser-centred crop the mask was found in. Provenance: a
    # suspect prediction can be re-examined in the exact window the model saw,
    # and a change of crop size becomes visible in the data rather than only in
    # the code that produced it.
    crop_x: int | None = Field(default=None)
    crop_y: int | None = Field(default=None)

    # The laser label that selected the fish. Named so the cohort can select on
    # *mismatch* rather than absence: a prediction derived from a dot that
    # RANSAC later supersedes is stale, and without this it would sit unnoticed
    # forever. Same lesson as `LaserDepth`'s provenance columns.
    laser_label_id: int | None = Field(default=None, foreign_key="laserlabel.id")

    # Which version of the head/tail-predict stage produced this row. The
    # cohort selects on a *mismatch* with the current version, so improving the
    # stage makes re-prediction an ordinary drainable cohort instead of a
    # hand-run backfill. NULL means "predates versioning", which reads as stale
    # exactly once. See `fishsense_shared.headtail_predictor` — and note this
    # stage needs it more than the laser one did: its mask backend and its crop
    # size are both behaviour a checkpoint hash would miss entirely.
    predictor_version: int | None = Field(default=None, index=True)
    # Recorded, never gated on — the same role `LaserDepth.residual_m` plays.
    # These answer "why did this frame come out that way" later;
    # `predictor_version` is the one thing anything decides on.
    checkpoint: str | None = Field(default=None)
    core_version: str | None = Field(default=None)

    # "predicted" | "no_detections" | "laser_off_all_fish" | "headtail_failed".
    # An abstention is a row, not a missing row — the cohort selects on absence,
    # so writing nothing would re-predict the same image forever, and the three
    # abstention kinds are different facts worth telling apart.
    status: str = Field(default="predicted")

    # The silhouette gate declined to seed this one. Distinct from an ordinary
    # abstention: the model did find a fish, and the row is kept so the band can
    # be retuned. Mirrors `LaserPrediction.rejected_out_of_region`.
    rejected_low_confidence: bool = Field(default=False)

    created_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)

    image_id: int | None = Field(default=None, foreign_key="image.id")
