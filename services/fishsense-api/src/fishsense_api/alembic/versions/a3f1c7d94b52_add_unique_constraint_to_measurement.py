"""add unique constraint to measurement on (image_id, fish_id)

One length per fish per frame. A frame can hold several fish, so the key
is the pair, not `image_id` alone.

This is the DB-level backstop for stage 14 re-run safety. Previously
`post_measurement` did `session.merge` on a body with `id=None`, which
keys on the primary key only and therefore always INSERTed — so a re-run
against a partially-measured dive duplicated every already-measured
image. The endpoint now upserts on the natural key; this constraint is
what refuses the duplicate even if some other writer skips that path.

Safe to apply without a dedup backfill: verified against prod before
writing this revision — 242 measurement rows, 0 duplicate
(image_id, fish_id) pairs, and 0 NULLs in either column. (NULLs would
matter because postgres treats them as distinct, so a row with a NULL
member would slip past the constraint.)

Revision ID: a3f1c7d94b52
Revises: 7934e62a12c0
Create Date: 2026-07-17 03:20:00.000000

"""
# pylint: skip-file

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a3f1c7d94b52"
down_revision: Union[str, Sequence[str], None] = "7934e62a12c0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_unique_constraint(
        "uq_measurement_image_fish", "measurement", ["image_id", "fish_id"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("uq_measurement_image_fish", "measurement", type_="unique")
