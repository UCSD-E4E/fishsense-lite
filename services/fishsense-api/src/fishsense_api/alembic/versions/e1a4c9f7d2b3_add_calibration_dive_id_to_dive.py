"""add calibration_dive_id self-link to dive

Lets a dive borrow another dive's laser calibration. Laser calibration is
physically a property of the camera+laser rig, not the dive, so a fish-only
dive with no slate frames can point at a sibling slate/calibration dive shot
with the same rig. NULL (the default) means "self-calibrate from my own slate
labels".

Revision ID: e1a4c9f7d2b3
Revises: d4e9a1c7b520
Create Date: 2026-07-22 00:00:00.000000

"""
# pylint: skip-file

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e1a4c9f7d2b3"
down_revision: Union[str, Sequence[str], None] = "d4e9a1c7b520"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "dive",
        sa.Column(
            "calibration_dive_id",
            sa.Integer(),
            sa.ForeignKey("dive.id", name="fk_dive_calibration_dive_id"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("fk_dive_calibration_dive_id", "dive", type_="foreignkey")
    op.drop_column("dive", "calibration_dive_id")
