"""add fish_angle_degrees to specieslabel

Revision ID: a1d47f2c9b83
Revises: e07c31b9a4d2
Create Date: 2026-09-09 00:00:00.000000

The reviewed angle of the fish, in degrees. `fish_angle_category` already
exists but cannot hold it: it is the Label Studio taxonomy leaf, and its top
choice is the open interval `x > 15°`. On dive 87
(`083123_Fish Deg Angle Tests_FSL06`) five consecutive groups are five
different commanded angles that every one of them records as `x > 15°`, so the
column collapses the experiment's independent variable into a single label.

Kept as a second column rather than a widening of the first because the two
have different writers. `fish_angle_category` arrives from the labeler through
`sync_species_labels_for_label_studio_project_activity`, which is its single
writer; `fish_angle_degrees` is entered by an operator reviewing a group and
must survive every sync pass untouched. Merging them would put the sync in
contention with hand-reviewed data.

**Nullable, and deliberately not backfilled** — the opposite call from
`c7e4a91f2d38`'s `needs_reprocess`, which needed `server_default=false` because
its NULL would have made `WHERE NOT needs_reprocess` skip every pre-existing
row under three-valued logic. Nothing here selects on the angle, so a NULL is
inert; and a `server_default` of 0.0 would be actively wrong, because 0° is a
real value in the sweep and would become indistinguishable from "nobody has
reviewed this yet".

Float rather than integer so a later test using finer steps than 5° does not
need another migration.

No dialect guard: `add_column` is plain DDL on both Postgres and the SQLite
the migration tests run against.
"""

# pylint: skip-file

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1d47f2c9b83"
down_revision: Union[str, Sequence[str], None] = "e07c31b9a4d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "specieslabel"
COLUMN = "fish_angle_degrees"


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(TABLE, sa.Column(COLUMN, sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column(TABLE, COLUMN)
