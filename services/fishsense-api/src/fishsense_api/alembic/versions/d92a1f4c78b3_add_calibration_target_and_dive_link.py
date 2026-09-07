"""add calibrationtarget and dive.calibration_target_id

A `CalibrationTarget` is a planar target of known geometry that laser
extrinsics can be fitted against. `DiveSlate` was the only one; most of the
pool-test corpus was shot against a **checkerboard**, which is not a slate and
should not be called one in the schema. See `models/calibration_target.py` and
`docs/plans/checkerboard-laser-calibration.md`.

**No row is seeded here.** The table is useless without `square_size_m`, and
that number is a caliper measurement of a physical board, not something a
migration can know. Seeding a nominal pitch off the board's PDF would be the
`fishmodelreference` Ruler mistake repeated — assumed 355.6 mm from its
nominal 14 in, actually 342.9 mm, a 3.3% scale error — except worse, because
scale error is precisely the term the reprojection residual cannot see, so it
would calibrate cleanly and measure every fish in the dive wrong. An empty
table means those dives simply stay uncalibrated, which is the failure
direction they are already in.

Revision ID: d92a1f4c78b3
Revises: c4f8a2e60b17
Create Date: 2026-09-07 00:00:00.000000

"""

# pylint: skip-file

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d92a1f4c78b3"
down_revision: Union[str, Sequence[str], None] = "c4f8a2e60b17"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Idempotent against `SQLModel.metadata.create_all`: `lifespan` runs
    `create_all` before alembic, so on a fresh database the table and the
    column already exist by the time this runs, and a bare `create_table` /
    `add_column` would raise and stop the API from starting.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "calibrationtarget" not in inspector.get_table_names():
        op.create_table(
            "calibrationtarget",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(length=100), nullable=False),
            # Interior corners, NOT squares: a 15 x 11 board has 14 x 10.
            sa.Column("rows", sa.Integer(), nullable=False),
            sa.Column("cols", sa.Integer(), nullable=False),
            # NOT NULL deliberately — see the module docstring.
            sa.Column("square_size_m", sa.Float(), nullable=False),
            sa.Column("notes", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("name", name="uq_calibrationtarget_name"),
        )
        op.create_index(
            "ix_calibrationtarget_name", "calibrationtarget", ["name"], unique=True
        )

    if "calibration_target_id" not in {
        c["name"] for c in inspector.get_columns("dive")
    }:
        op.add_column(
            "dive",
            sa.Column("calibration_target_id", sa.Integer(), nullable=True),
        )
        # Named so the downgrade and any future ALTER can find it. Postgres
        # only — SQLite renders foreign keys inline at CREATE TABLE and cannot
        # add one afterwards, and the migration tests run on SQLite.
        if bind.dialect.name == "postgresql":
            op.create_foreign_key(
                "fk_dive_calibration_target_id",
                "dive",
                "calibrationtarget",
                ["calibration_target_id"],
                ["id"],
            )


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "calibration_target_id" in {c["name"] for c in inspector.get_columns("dive")}:
        if bind.dialect.name == "postgresql":
            op.drop_constraint(
                "fk_dive_calibration_target_id", "dive", type_="foreignkey"
            )
        op.drop_column("dive", "calibration_target_id")

    if "calibrationtarget" in inspector.get_table_names():
        op.drop_index("ix_calibrationtarget_name", table_name="calibrationtarget")
        op.drop_table("calibrationtarget")
