"""add width/height to laserprediction

The laser populate step converts a prediction's pixel x/y into the percentages
Label Studio keypoints use, which needs the rectified frame dimensions the
prediction was made against. The GPU predict stage now records them.

Revision ID: b8e3f1a09d24
Revises: a7d21e4f0c93
Create Date: 2026-07-28 00:00:00.000000

"""
# pylint: skip-file

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b8e3f1a09d24"
down_revision: Union[str, Sequence[str], None] = "a7d21e4f0c93"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("laserprediction", sa.Column("width", sa.Integer(), nullable=True))
    op.add_column("laserprediction", sa.Column("height", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("laserprediction", "height")
    op.drop_column("laserprediction", "width")
