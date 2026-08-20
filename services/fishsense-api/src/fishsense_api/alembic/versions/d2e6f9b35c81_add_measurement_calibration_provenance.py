"""add measurement.laser_extrinsics_id

Which calibration a length was computed with. A fish length is a function of
the `LaserExtrinsics` behind its depth, and extrinsics get replaced — the
2026-08-11 slate panel-offset fix recalibrated 6 of the 8 dives that already
had measurements, silently invalidating their lengths while
`dive_pipeline_status.measured` still read true and the stage-14 cohort kept
skipping them.

Without provenance the only options were to rewrite every measurement on
every run (churning values nothing had shown to be wrong) or to leave stale
ones in place forever. With it, stage 14 skips an image only when its
measurement names the extrinsics row the dive resolves to *today*.

Backfill note: every existing row gets NULL, which never matches a resolved
id, so the entire pre-provenance backlog reads as stale and re-enters the
cohort once. That is deliberate — those lengths predate at least one
recalibration — and it drains at the hourly stage-14 cadence, one dive per
firing, stamping provenance as it goes.

Revision ID: d2e6f9b35c81
Revises: c1d5e8a24b70
Create Date: 2026-08-18 00:00:00.000000

"""
# pylint: skip-file

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d2e6f9b35c81"
down_revision: Union[str, Sequence[str], None] = "c1d5e8a24b70"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Idempotent against `SQLModel.metadata.create_all` — see the laserdepth
    revision. Add the column only when it is missing.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {c["name"] for c in inspector.get_columns("measurement")}
    if "laser_extrinsics_id" in existing:
        return
    op.add_column(
        "measurement", sa.Column("laser_extrinsics_id", sa.Integer(), nullable=True)
    )
    op.create_foreign_key(
        "fk_measurement_laser_extrinsics_id",
        "measurement",
        "laserextrinsics",
        ["laser_extrinsics_id"],
        ["id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "fk_measurement_laser_extrinsics_id", "measurement", type_="foreignkey"
    )
    op.drop_column("measurement", "laser_extrinsics_id")
