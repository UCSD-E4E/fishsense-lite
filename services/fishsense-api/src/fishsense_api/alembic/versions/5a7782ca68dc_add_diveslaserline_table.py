"""add diveslaserline table

Revision ID: 5a7782ca68dc
Revises: fab962df0484
Create Date: 2026-08-01 15:14:02.480958

"""

# pylint: skip-file

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "5a7782ca68dc"
down_revision: Union[str, Sequence[str], None] = "fab962df0484"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Idempotent against `SQLModel.metadata.create_all`: the FastAPI lifespan
    runs `create_all` before alembic upgrade, and `divelaserline` is in the ORM
    model registry, so on an existing DB the table already exists by the time
    this migration runs. Skip the DDL when the table is present.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("divelaserline"):
        return
    op.create_table(
        "divelaserline",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dive_id", sa.Integer(), nullable=True),
        sa.Column("a", sa.Float(), nullable=False),
        sa.Column("b", sa.Float(), nullable=False),
        sa.Column("c", sa.Float(), nullable=False),
        sa.Column("n_points", sa.Integer(), nullable=False),
        sa.Column("inlier_count", sa.Integer(), nullable=False),
        sa.Column("inlier_fraction", sa.Float(), nullable=False),
        sa.Column("residual_std", sa.Float(), nullable=False),
        sa.Column("label_noise_mad", sa.Float(), nullable=False),
        sa.Column("line_confidence", sa.Float(), nullable=False),
        sa.Column(
            "fitted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["dive_id"], ["dive.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dive_id", name="uq_divelaserline_dive_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("divelaserline")
