"""add colour, out-of-region and stage-version fields to laserprediction

Revision ID: d81b6c4a5f27
Revises: c7e4a91f2d38
Create Date: 2026-08-28 00:00:00.000000

Two additions to the model-assisted laser stage.

`color` / `color_margin` record the laser's colour, read off the dot's own
pixels. The pre-annotation used to hardcode "Red Laser", which is wrong for
about a quarter of prod: 143 dives are entirely red and 88 entirely green, and
nothing was tracking which. Populate takes the dive-level majority of these,
because colour is a property of the rig for the whole dive.

`rejected_out_of_region` marks a frame where the detector *did* find a dot but
it fell outside the expected-laser region and was dropped. It exists to keep
that distinguishable from an ordinary non-detection: without it, a region that
was cut too tight would look exactly like a model that had stopped finding
lasers, and there would be nothing in the data to tell the two apart.

`predictor_version` is what makes improving the detector a drainable cohort
rather than a hand-run backfill: the stage selects on a *mismatch* with the
current version, the same way the laser-depth and measurement cohorts select
on a mismatch with the calibration that produced them. It is deliberately
nullable — the 1,555 rows already in prod predate versioning, and NULL is
exactly "unknown, therefore stale", which drains on the first pass. Backfilling
them to 1 would be a guess about behaviour nobody recorded.

`checkpoint` / `core_version` are recorded and never gated on, for answering
"why did this frame come out that way" after the fact.

`server_default=false` + NOT NULL on the boolean for the same reason as
`needs_reprocess` in c7e4a91f2d38 — a NULL backfill makes
`WHERE NOT rejected_out_of_region` silently drop every pre-existing row under
three-valued logic. The two colour columns are genuinely nullable: NULL there
means "no opinion", which is a state the classifier really has (no dot to
sample, or channels too close to call) and must not be confused with "red".
"""

# pylint: skip-file

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d81b6c4a5f27"
down_revision: Union[str, Sequence[str], None] = "c7e4a91f2d38"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "laserprediction"


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(TABLE, sa.Column("color", sa.String(), nullable=True))
    op.add_column(TABLE, sa.Column("predictor_version", sa.Integer(), nullable=True))
    op.add_column(TABLE, sa.Column("checkpoint", sa.String(), nullable=True))
    op.add_column(TABLE, sa.Column("core_version", sa.String(), nullable=True))
    op.create_index(
        "ix_laserprediction_predictor_version", TABLE, ["predictor_version"]
    )
    op.add_column(TABLE, sa.Column("color_margin", sa.Float(), nullable=True))
    op.add_column(
        TABLE,
        sa.Column(
            "rejected_out_of_region",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_laserprediction_predictor_version", table_name=TABLE)
    op.drop_column(TABLE, "core_version")
    op.drop_column(TABLE, "checkpoint")
    op.drop_column(TABLE, "predictor_version")
    op.drop_column(TABLE, "rejected_out_of_region")
    op.drop_column(TABLE, "color_margin")
    op.drop_column(TABLE, "color")
