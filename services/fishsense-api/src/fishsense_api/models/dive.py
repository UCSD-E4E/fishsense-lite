"""Model representing a dive."""

from datetime import datetime

from sqlmodel import Column, DateTime, Enum, Field

from fishsense_api.models.model_base import ModelBase
from fishsense_api.models.priority import Priority


class Dive(ModelBase, table=True):
    """Model representing a dive."""

    id: int | None = Field(default=None, primary_key=True)
    name: str | None = Field(default=None, index=True)
    path: str = Field(max_length=255, unique=True, index=True)
    dive_datetime: datetime = Field(sa_type=DateTime(timezone=True), default=None)
    priority: Priority = Field(default=Priority.LOW, sa_column=Column(Enum(Priority)))
    flip_dive_slate: bool | None = Field(default=False)

    # Free-text operator note. Exists to carry the *reason* a dive is in an
    # unusual state — overwhelmingly, why it was set `Priority.NONE`. Nothing
    # in the pipeline reads it: it is for the human looking at a dive that
    # will never drain and asking why. Deliberately unstructured, because the
    # reasons are one-offs (a slate with no scan, a corrupted card, a dive
    # shot on a rig whose calibration was never recoverable) and any schema
    # we invented for them would be wrong by the third case.
    notes: str | None = Field(default=None)

    camera_id: int | None = Field(default=None, foreign_key="camera.id")
    dive_slate_id: int | None = Field(default=None, foreign_key="diveslate.id")

    # Which planar calibration target this dive was shot against, when it was
    # not one of the `DiveSlate` templates. Set by the species-label sync from
    # the `Calibration Targets` taxonomy branch, and by an operator.
    #
    # Independent of `dive_slate_id` rather than an alternative spelling of
    # it: the two name different physical objects and gate different stages,
    # and a dive that carries a slate keeps calibrating through stage 13. NULL
    # is the overwhelmingly common case — only the pool-test calibration dives
    # were shot against a checkerboard.
    calibration_target_id: int | None = Field(
        default=None, foreign_key="calibrationtarget.id"
    )

    # Self-referential link to the dive whose laser calibration this dive
    # borrows. Laser calibration is physically a property of the camera+laser
    # rig, not the dive, so a dive with no slate frames of its own (e.g. a
    # fish-only dive) can point at a sibling slate/calibration dive shot with
    # the same rig. When set, laser-extrinsics resolution and the
    # `calibrated` gate fall back to this dive's LaserExtrinsics. NULL means
    # "self-calibrate from my own slate labels" (the default).
    calibration_dive_id: int | None = Field(default=None, foreign_key="dive.id")

    # When a calibration fit was last REFUSED for this dive, and why.
    #
    # Both calibration cohorts select on dive *state* -- "has no usable
    # `LaserExtrinsics` row" -- and a refusal does not change that state. So
    # without this a dive whose observations cannot produce a sound fit is
    # re-selected every hour forever, re-staging its raw `.ORF`s from the NAS
    # each time, and because the selectors are `ORDER BY id LIMIT 1` it blocks
    # every dive behind it. That is the prod dive-347 shape, and the baseline
    # gate made it reachable for eight more dives at once.
    #
    # Only DETERMINISTIC refusals are recorded: too few observations, a fit
    # that disagrees with its own dots, an implausible baseline. Those are
    # functions of the observations the run was dispatched with, so retrying
    # re-derives them. A transient failure (NAS unreachable, worker evicted)
    # must NOT land here, or a blip would park a healthy dive.
    #
    # **It expires on its own.** The cohorts ignore a refusal once any laser or
    # slate label on the dive has been updated more recently, so relabelling
    # brings the dive straight back without an operator touching anything. That
    # is what keeps this from becoming a permanent exclusion nobody remembers
    # setting -- the failure mode a bare boolean would have.
    calibration_refused_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True)
    )
    calibration_refused_reason: str | None = Field(default=None)

    # The newest label timestamp the refused fit was computed from.
    #
    # **Expiry compares this, NOT `calibration_refused_at`, and the difference
    # is a silent data-loss bug.** Label timestamps come from Label Studio
    # (the sync copies `task.updated_at` verbatim); `calibration_refused_at` is
    # the API's own wall clock. Comparing the two mixes clocks: a labeler who
    # fixes a dive at 10:20 is only synced at the top of the hour, so against a
    # 10:50 refusal their corrected label looks OLDER and the dive stays
    # excluded forever — and the ~50 minutes between each hourly sync and the
    # +50/+52 calibration slots is exactly when a labeler responding to a
    # wedged dive works.
    #
    # Storing the max label timestamp at refusal time keeps both sides of the
    # comparison in Label Studio's clock, so "a label is newer than what we
    # fitted from" means what it says. NULL means the dive had no labels then,
    # so any label at all is newer.
    calibration_refused_labels_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True)
    )
