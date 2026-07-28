"""add laserprediction table

Model-predicted laser dots from the fishsense-core LaserDetector stage. One
row per image (unique image_id — re-prediction upserts). Read by laser
populate to seed Label Studio pre-annotations (assisted review); kept separate
from LaserLabel so predictions never count toward the human "valid laser" gate.

Revision ID: a7d21e4f0c93
Revises: f2b5d0c8e3a1
Create Date: 2026-07-28 00:00:00.000000

"""
# pylint: skip-file

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7d21e4f0c93"
down_revision: Union[str, Sequence[str], None] = "f2b5d0c8e3a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Idempotent against `SQLModel.metadata.create_all`: the FastAPI
    lifespan runs `create_all` before alembic upgrade, and
    `laserprediction` is in the ORM model registry, so on an existing
    DB the table already exists by the time this migration runs. Skip
    the DDL when the table is present rather than crash startup with
    `DuplicateTableError`.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("laserprediction"):
        return
    op.create_table(
        "laserprediction",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("x", sa.Float(), nullable=True),
        sa.Column("y", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("image_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["image_id"], ["image.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("image_id", name="uq_laser_prediction_image"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("laserprediction")
