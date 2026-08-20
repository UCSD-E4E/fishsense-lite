"""Measurement model for the FishSense API."""

from sqlmodel import Field, SQLModel, UniqueConstraint


class Measurement(SQLModel, table=True):
    """Measurement model representing fish measurements in the database."""

    # One length per fish per frame. A frame can hold several fish, so the
    # key is the pair — not `image_id` alone. This is the DB-level backstop
    # for stage 14's re-run safety; `post_measurement` upserts on the same
    # pair, and the activity skips already-measured images before it gets
    # here. Both layers are deliberate: the filter avoids pointless work,
    # this refuses to record the duplicate even if the filter is bypassed.
    __table_args__ = (
        UniqueConstraint(
            "image_id",
            "fish_id",
            name="uq_measurement_image_fish",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    length_m: float | None = Field(default=None)

    image_id: int | None = Field(default=None, foreign_key="image.id")
    fish_id: int | None = Field(default=None, foreign_key="fish.id")
    # Which calibration produced this length. A length is only meaningful
    # relative to the extrinsics behind its depth, and extrinsics do get
    # replaced (the 2026-08-11 slate panel-offset fix recalibrated 6 of the 8
    # dives that already had measurements). Recording it is what lets stage
    # 14 ask "is this still current?" instead of choosing between rewriting
    # every row on every run and never revisiting a stale one; the cohort
    # selector re-picks a dive exactly when this disagrees with the
    # extrinsics that would be resolved today. NULL on every row written
    # before this column existed, which reads as stale — deliberately, since
    # those lengths predate at least one recalibration.
    laser_extrinsics_id: int | None = Field(
        default=None, foreign_key="laserextrinsics.id"
    )
