"""laserextrinsics dedup+created_at default+unique dive_id

Revision ID: fab962df0484
Revises: f4a9c2b17e60
Create Date: 2026-08-01 14:33:11.611031

"""

# pylint: skip-file

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "fab962df0484"
down_revision: Union[str, Sequence[str], None] = "f4a9c2b17e60"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Forward-fix for the #462-recovery bug (see LaserExtrinsics model docstring):
    the calibration path appended rows and left `created_at` NULL, which broke
    the latest-wins read. Order matters — de-dup and backfill the data first,
    then the unique constraint can be added without a violation.
    """
    # 1. De-duplicate: keep the newest row per dive (highest id; created_at may
    #    be NULL on buggy rows so id is the reliable tiebreaker).
    op.execute("""
        DELETE FROM laserextrinsics a
        USING laserextrinsics b
        WHERE a.dive_id = b.dive_id
          AND a.dive_id IS NOT NULL
          AND a.id < b.id
        """)
    # 2. Backfill any NULL created_at (the hand-patched #462-recovery rows
    #    already have timestamps; this covers anything left).
    op.execute("UPDATE laserextrinsics SET created_at = now() WHERE created_at IS NULL")
    # 3. Column default so future inserts can never be NULL again.
    op.alter_column(
        "laserextrinsics",
        "created_at",
        server_default=sa.text("now()"),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=True,
    )
    # 4. One calibration per dive.
    op.create_unique_constraint(
        "uq_laserextrinsics_dive_id", "laserextrinsics", ["dive_id"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("uq_laserextrinsics_dive_id", "laserextrinsics", type_="unique")
    op.alter_column(
        "laserextrinsics",
        "created_at",
        server_default=None,
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=True,
    )
