"""dive_pipeline_status: `measured` requires the current calibration

`measured` asked only whether a Measurement row existed. That made it blind to
the failure it most needed to catch: replacing a dive's calibration
invalidates every length computed from the old one, and the 2026-08-11 slate
panel-offset fix did exactly that to 6 of the 8 dives that already had
measurements. The flag stayed true, the stage-14 cohort (which mirrors this
predicate) kept skipping them, and the stale lengths stood.

An image now counts as measured only when its Measurement names the
`LaserExtrinsics` row the dive resolves to today — its own, else the one
borrowed through `calibration_dive_id`, matching
`get_laser_extrinsics_for_dive`.

Prod effect: every measurement predating `measurement.laser_extrinsics_id`
carries NULL, so `measured` reads false for those dives until stage 14
revisits them. That is a real state change on the Superset dashboards, and it
is the honest reading — those lengths are unverified against the current
calibration. It self-clears at the hourly stage-14 cadence.

Drop + recreate rather than CREATE OR REPLACE: Postgres is restrictive about
column-shape changes on replace, and the view has no dependents.

Revision ID: e3f70ac46d92
Revises: d2e6f9b35c81
Create Date: 2026-08-18 00:00:00.000000

"""
# pylint: skip-file

from typing import Sequence, Union

from alembic import op

from fishsense_api.views import (
    DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
)

# revision identifiers, used by Alembic.
revision: str = "e3f70ac46d92"
down_revision: Union[str, Sequence[str], None] = "d2e6f9b35c81"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Recreate the view with the calibration-provenance requirement."""
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)


def downgrade() -> None:
    """Recreate from whatever `views.py` currently holds.

    Not a true inverse — the SQL is imported from the single source of truth,
    so a downgrade after reverting the code restores the old predicate and a
    downgrade without reverting is a no-op. Same trade-off every view
    migration in this tree makes.
    """
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)
