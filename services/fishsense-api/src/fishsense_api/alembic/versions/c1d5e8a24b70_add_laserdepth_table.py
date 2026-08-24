"""add laserdepth table

Per-image distance to the laser dot. Stage 14 has always computed this en
route to a fish length — `laser3d = compute_world_point_from_laser(...)`, then
`depth = laser3d[2]` — and discarded it, so the number existed only for the
measurable frames stage 14 visits and was never queryable. One row per image
(unique image_id, recompute upserts).

Two distances because they are different numbers: `depth_m` is the Z
component, the depth stage 14 back-projects head and tail against, and
`range_m` is the Euclidean norm, the true slant distance to the dot.
`residual_m` is the closest-approach distance between the camera ray and the
laser ray — ~0 when the dot really is consistent with the calibration —
recorded as a quality signal, not gated on.

`laser_label_id` and `laser_extrinsics_id` record the two inputs. Either
changing invalidates the depth — a relabel moves the dot, a recalibration
moves the ray — and `select_next_for_laser_depth` re-picks a dive on exactly
that mismatch, which is what makes a recompute a drainable cohort rather than
a hand-run backfill.

Revision ID: c1d5e8a24b70
Revises: a2f7c31d9e64
Create Date: 2026-08-18 00:00:00.000000

"""

# pylint: skip-file

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c1d5e8a24b70"
down_revision: Union[str, Sequence[str], None] = "a2f7c31d9e64"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Idempotent against `SQLModel.metadata.create_all`: the FastAPI lifespan
    runs `create_all` before alembic upgrade and `laserdepth` is in the ORM
    model registry, so on an existing DB the table is already there by the
    time this runs. Skip rather than crash startup with `DuplicateTableError`.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("laserdepth"):
        return
    op.create_table(
        "laserdepth",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("depth_m", sa.Float(), nullable=False),
        sa.Column("range_m", sa.Float(), nullable=True),
        sa.Column("residual_m", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("image_id", sa.Integer(), nullable=True),
        sa.Column("laser_label_id", sa.Integer(), nullable=True),
        sa.Column("laser_extrinsics_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["image_id"], ["image.id"]),
        sa.ForeignKeyConstraint(["laser_label_id"], ["laserlabel.id"]),
        sa.ForeignKeyConstraint(["laser_extrinsics_id"], ["laserextrinsics.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("image_id", name="uq_laser_depth_image"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("laserdepth")
