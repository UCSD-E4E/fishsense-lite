"""add needs_reprocess to the four label tables

Revision ID: c7e4a91f2d38
Revises: b3d5e91a7c42
Create Date: 2026-08-27 00:00:00.000000

Marks a label whose image must have its overlay JPEG regenerated. The
preprocess cohorts all select on "image has no label row of this kind", so an
image drops out the moment a row exists and its processed JPEG is frozen from
then on; this flag is how a redraw gets requested without deleting the label
and throwing away a labeler's work.

One column per label table rather than one on `image`, because an image
carries a different JPEG per stage (`preprocess_jpeg`,
`preprocess_groups_jpeg`, `preprocess_headtail_jpeg`,
`preprocess_slate_images_jpeg`) and a change to one stage's overlay says
nothing about the other three.

`server_default="false"` matters for the backfill: without it every existing
row lands NULL, and `WHERE NOT needs_reprocess` silently drops NULL rows under
three-valued logic — the same shape as the `created_at IS NULL` bug that made
`ORDER BY created_at DESC` match nothing on `laserextrinsics`. The column is
NOT NULL for the same reason.

No dialect guard needed here (unlike `b3d5e91a7c42`'s enum): `add_column` is
plain DDL on both Postgres and the SQLite the migration tests run against.
And no `IF NOT EXISTS` dance either — `run_alembic_upgrade` *stamps* head on a
fresh database rather than replaying migrations over the `create_all` schema,
so this only ever runs against a database that predates the column.
"""

# pylint: skip-file

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7e4a91f2d38"
down_revision: Union[str, Sequence[str], None] = "b3d5e91a7c42"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

LABEL_TABLES = ("laserlabel", "specieslabel", "headtaillabel", "diveslatelabel")


def upgrade() -> None:
    """Upgrade schema."""
    for table in LABEL_TABLES:
        op.add_column(
            table,
            sa.Column(
                "needs_reprocess",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )


def downgrade() -> None:
    """Downgrade schema."""
    for table in LABEL_TABLES:
        op.drop_column(table, "needs_reprocess")
