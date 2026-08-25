"""Mark estimated reference lengths, and repair the notes the seed could miss.

Three things, all consequences of `Weasly Fish` being seeded at a PROVISIONAL
length in `d1a7b3e95c02`.

**1. `is_provisional`.** `fish_model_species_mislabel_suspects` CROSS JOINs
`fishmodelreference` to ask "does this frame fit some OTHER model better?". A
length nobody has calipered must not answer that question: stage 14 measures a
projection, so angled frames of the big models read short and never long, and an
estimate dropped into that short band flags every correct-but-angled frame near
it. Measured: a Shark (605 mm) at 290 mm had no suspect before — closest other
reference was the Ruler, 15.4% away, outside the 10% gate — and lands at 3.3%
from a 300 mm Weasly Fish. The view now excludes provisional rows; they stay
fully graded in `fish_model_measurement_accuracy`.

**2. Notes backfill.** `e2c9a4f70b31` and `b4c81f60d7e2` iterate the *live*
`KNOWN_FISH_MODELS`, so on any database below them they now insert Weasly Fish
themselves — with `notes = NULL`, since their INSERT binds only
(name, known_length_m). `d1a7b3e95c02` then sees the row present and
short-circuits, leaving the caliper provenance permanently missing. Filling a
NULL is safe; a non-NULL note is an operator's and is left alone.

**3. View rebuild**, because the mislabel SQL changed.

Idempotent against `create_all`: the column exists already on a fresh database
(it is on the model), so the ALTER is skipped.

Revision ID: f2b8c04e71a3
Revises: d1a7b3e95c02
"""

# pylint: skip-file
# See d1a7b3e95c02 — alembic's generated module shape trips invalid-name and
# no-member across all migrations here.

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from fishsense_api.views import (
    DROP_FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL,
    FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL,
    FISH_MODEL_NOTES,
    PROVISIONAL_FISH_MODELS,
)

revision: str = "f2b8c04e71a3"
down_revision: Union[str, Sequence[str], None] = "d1a7b3e95c02"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(bind, table: str, column: str) -> bool:
    inspector = sa.inspect(bind)
    if not inspector.has_table(table):
        return False
    return column in {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    """Add the column, mark the provisional rows, backfill missing notes."""
    bind = op.get_bind()

    if not _has_column(bind, "fishmodelreference", "is_provisional"):
        op.add_column(
            "fishmodelreference",
            sa.Column(
                "is_provisional",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )

    # Set from the declared set rather than a literal, so adding a provisional
    # model is one edit in views.py. Names absent from the table are simply not
    # matched — no failure if the seed never ran.
    for name in sorted(PROVISIONAL_FISH_MODELS):
        bind.execute(
            sa.text(
                "UPDATE fishmodelreference SET is_provisional = :flag "
                "WHERE name = :name"
            ),
            {"flag": True, "name": name},
        )

    # Fill only where a note is genuinely absent — never overwrite one an
    # operator wrote.
    for name, note in FISH_MODEL_NOTES.items():
        bind.execute(
            sa.text(
                "UPDATE fishmodelreference SET notes = :note "
                "WHERE name = :name AND notes IS NULL"
            ),
            {"note": note, "name": name},
        )

    op.execute(DROP_FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL)
    op.execute(FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL)


def downgrade() -> None:
    """Drop the column and rebuild the view without the provisional filter.

    The reference ROWS are left alone — they are data, and a downgrade that
    discards a length or a caliper note would lose work the upgrade path went
    out of its way to protect.
    """
    bind = op.get_bind()

    # Drop the view rather than rebuilding it by editing its SQL here: the
    # current definition references the column being removed, and reconstructing
    # the old text by string surgery would silently rot the moment the view
    # changes for any other reason. A downgrade is paired with a code revert,
    # and `run_alembic_upgrade`'s missing-view repair recreates every view from
    # whatever code is actually deployed on the next start.
    op.execute(DROP_FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL)

    if _has_column(bind, "fishmodelreference", "is_provisional"):
        op.drop_column("fishmodelreference", "is_provisional")
