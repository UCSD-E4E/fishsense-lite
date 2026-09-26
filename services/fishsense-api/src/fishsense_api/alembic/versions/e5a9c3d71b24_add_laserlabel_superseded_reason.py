"""Record why a laser label was superseded.

`laserlabel.superseded` is a dead letter with no provenance, so when the laser
validator was found to have eroded dives a run at a time nothing on the row
could tell its supersedes from an operator's deliberate one. From here on every
writer says who it was.

Nullable, no backfill: a pre-existing supersede's reason is genuinely unknown
and the migration does not guess. A plain VARCHAR rather than a Postgres
native enum, so a new reason is a code change and not an
`ALTER TYPE ... ADD VALUE` (see `b3d5e91a7c42`).

Revision ID: e5a9c3d71b24
Revises: c4e8b17a205f
"""

# pylint: skip-file
#   `alembic.op` is assembled at runtime via a proxy, so pylint sees no
#   `add_column` / `drop_column` members. Every migration in this directory
#   carries the same pragma.

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e5a9c3d71b24"
down_revision: Union[str, Sequence[str], None] = "c4e8b17a205f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add `laserlabel.superseded_reason`."""
    op.add_column(
        "laserlabel",
        sa.Column("superseded_reason", sa.String(length=40), nullable=True),
    )


def downgrade() -> None:
    """Drop `laserlabel.superseded_reason`."""
    op.drop_column("laserlabel", "superseded_reason")
