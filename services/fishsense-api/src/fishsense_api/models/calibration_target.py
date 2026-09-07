"""A planar target of known geometry that laser extrinsics can be fitted to.

Stage 13 has always fitted `LaserExtrinsics` from a `DiveSlate`, but the
slate's *only* contribution to that fit is a plane: `calibration_geometry`
takes correspondences, recovers a pose, and intersects the laser dot's
back-projected ray with the plane it implies. Nothing downstream — the
Atanasov fit, the self-consistency gate, `LaserExtrinsics`, stage 14 — learns
what the target was.

Most of the pool-test corpus was shot against a **checkerboard** rather than a
slate, so those dives cannot calibrate today and fail silently doing it (no
`dive_slate_id` -> stage 9 never fires -> stage 13 returns None). This table is
how such a dive says what it was shot against. See
`docs/plans/checkerboard-laser-calibration.md`.

**A separate table rather than a `DiveSlate` row.** A checkerboard *is* a
planar grid of known points, and reusing `DiveSlate` would need no stage-13
change at all — but it would call a checkerboard a "dive slate" in the schema,
in Label Studio, and in `dive_pipeline_status.slate_*`. This codebase has been
bitten by exactly that shape of misnomer before (the drifted stage-13 comments,
the four `put_*_label` clones), and the honest table also gives
`square_size_m` somewhere to live as a first-class measured quantity rather
than as an inferred `dpi`.

**`rows` and `cols` are INTERIOR CORNERS, not squares.** A 15 x 11 board has
14 x 10 interior corners, and the corners are what a detector returns and what
`solvePnP` is given. Storing squares and subtracting one somewhere would be
one more place to get it wrong.
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import DateTime, Field

from fishsense_api.models.model_base import ModelBase


class CalibrationTarget(ModelBase, table=True):
    """A calibration target's identity and geometry, in metres."""

    id: int | None = Field(default=None, primary_key=True)

    # The join key to the Label Studio species taxonomy: species sync reads
    # the `Calibration Targets` branch and matches its leaf against this
    # column, exactly as the slate-type leaves match `DiveSlate.name`. So a
    # second board is a taxonomy choice plus a row whose name matches it.
    name: str = Field(max_length=100, unique=True, index=True)

    # Interior corners. See the module docstring — NOT the square count.
    rows: int = Field()
    cols: int = Field()

    # The grid pitch, in metres. Required, and deliberately so: it is the only
    # number that sets the scale of every length the dive ultimately produces,
    # and scale error is the one term the reprojection residual provably
    # cannot see (rho = -0.026 over 1109 depths). A nullable column would let
    # a target exist that reads as usable and silently is not.
    #
    # Measure it across many squares and divide — measuring one square
    # multiplies the reading error by the grid count. The `fishmodelreference`
    # Ruler is the standing warning: assumed 355.6 mm from its nominal 14 in,
    # actually 342.9 mm, a 3.3% scale error that took two tick-counts to
    # believe.
    square_size_m: float = Field()

    # Free text for how and when the pitch was measured, and by whom. The
    # number above is only as good as its provenance, and a board that cannot
    # be re-measured should be treated the way V-Slate 7 is: refuse to
    # calibrate rather than guess.
    notes: str | None = Field(default=None)

    created_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)
