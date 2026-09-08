"""seed the Box reference and make box frames measurable

Revision ID: a1c6d0f483b7
Revises: e9c1a7b40d53
Create Date: 2026-09-07 00:00:00.000000

`Box` joins `Ruler` under the species taxonomy's `Calibration Targets` branch:
a rigid target with a known 0.15 m span, measured through the same name-keyed
path as the fish models.

Two halves, and each is useless alone. The seed gives the row the accuracy view
INNER JOINs on — without it a box measurement is silently absent rather than
wrong, which is the Weasly Fish failure. The view recreate widens the measurable
predicate (`taxonomy.rigid_target_sql` now renders an `IN` over
`MEASURABLE_CALIBRATION_TARGETS`) so the stage-14 cohort and
`dive_pipeline_status.measured` offer box frames at all.

`E4E Checkerboard` shares that branch and stays OUT of the allowlist
deliberately: it spans no single head/tail distance, so widening to it would
put frames in the cohort that `measure_fish_activity` always skips — the
never-goes-false shape that blocked scheduling stage 14 before 2026-07-17.

`fish_model_species_mislabel_suspects` is recreated too, and that half is a
consequence of the seed rather than of the taxonomy. It CROSS JOINs
`fishmodelreference` to ask which model a frame's length fits best, and the Box
at 0.150 m is the first reference low enough to sit in the band that
foreshortened frames of the ~0.195 m models land in — a correct Gray Anthias
measuring 0.160 m would be flagged against a box. Calibration targets are now
excluded from both sides of that view (they are not candidate species labels),
which also retires the Ruler's smaller, pre-existing exposure.

The accuracy view is untouched, for the same reason the ruler's migration left
it alone: it joins `fishmodelreference` on `Fish.name`, so "Box" flows through
unchanged once the reference row exists — and calibration targets stay fully
graded there, which is the point of them.

"""

# pylint: skip-file

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from fishsense_api.views import (
    DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL,
    FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL,
    FISH_MODEL_NOTES,
    KNOWN_FISH_MODELS,
    PROVISIONAL_FISH_MODELS,
)

# revision identifiers, used by Alembic.
revision: str = "a1c6d0f483b7"
down_revision: Union[str, Sequence[str], None] = "e9c1a7b40d53"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Seed any missing reference rows, then widen the measurable predicate."""
    bind = op.get_bind()
    existing = {
        row[0] for row in bind.execute(sa.text("SELECT name FROM fishmodelreference"))
    }
    # Insert-only: an operator who has calipered a target and corrected its row
    # must not have that stamped back to the committed value.
    for model in KNOWN_FISH_MODELS:
        if model["name"] in existing:
            continue
        bind.execute(
            sa.text(
                "INSERT INTO fishmodelreference "
                "(name, known_length_m, notes, is_provisional) "
                "VALUES (:name, :known_length_m, :notes, :is_provisional)"
            ),
            {
                "name": model["name"],
                "known_length_m": model["known_length_m"],
                "notes": FISH_MODEL_NOTES.get(model["name"]),
                "is_provisional": model["name"] in PROVISIONAL_FISH_MODELS,
            },
        )

    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DROP_FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL)
    op.execute(FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL)


def downgrade() -> None:
    """Rebuild the view from `views.py`, and leave the seeded row alone.

    Note what this does NOT do: it cannot narrow the predicate, because the
    view SQL is generated from the live `views.py` constant, which by
    definition carries whatever `rigid_target_sql` currently renders. A real
    downgrade of the predicate means reverting the source too. What the
    recreate buys is a view that matches the code after an
    `upgrade`/`downgrade` round trip rather than one built from a half-applied
    state — the same posture as `b4c81f60d7e2`, whose docstring claimed the
    narrowing this one doesn't.

    The reference row stays: it is inert without the predicate, and dropping it
    would discard a span an operator may have corrected by hand.
    """
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DROP_FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL)
    op.execute(FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL)
