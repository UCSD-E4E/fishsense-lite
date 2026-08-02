"""add slateprediction table

Revision ID: a3d0440cda27
Revises: 5a7782ca68dc
Create Date: 2026-08-02 09:46:49.230547

"""
# pylint: skip-file

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3d0440cda27'
down_revision: Union[str, Sequence[str], None] = '5a7782ca68dc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Idempotent against `SQLModel.metadata.create_all`: the FastAPI lifespan runs
    `create_all` before alembic upgrade, and `slateprediction` is in the ORM
    model registry, so on an existing DB the table already exists by the time
    this migration runs. Skip the DDL when the table is present.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("slateprediction"):
        return
    op.create_table(
        "slateprediction",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("reference_points", sa.JSON(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("rejected_reason", sa.String(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("image_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["image_id"], ["image.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("image_id", name="uq_slate_prediction_image"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("slateprediction")
