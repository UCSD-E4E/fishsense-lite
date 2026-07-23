"""`calibrated` counts borrowed calibration via calibration_dive_id

The stage-13 `calibrated` flag checked only "dive has its own LaserExtrinsics
row". With the new `Dive.calibration_dive_id` self-link, a fish-only dive can
borrow a sibling slate dive's calibration, and `get_laser_extrinsics_for_dive`
+ the stage-14 cohort now resolve through that link. The view lagged, so a
linked dive read `calibrated = false` on the dashboard even though it was
measurable. `calibrated` now ORs in the linked source dive's extrinsics.

Drop + recreate rather than CREATE OR REPLACE — postgres is restrictive about
column-shape changes and the view has no dependents.

Revision ID: f2b5d0c8e3a1
Revises: e1a4c9f7d2b3
Create Date: 2026-07-22 00:05:00.000000

"""
# pylint: skip-file

from typing import Sequence, Union

from alembic import op

from fishsense_api.views import (
    DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
)

# revision identifiers, used by Alembic.
revision: str = "f2b5d0c8e3a1"
down_revision: Union[str, Sequence[str], None] = "e1a4c9f7d2b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)


def downgrade() -> None:
    """Drop the new view; the prior migration's upgrade is the recreate.
    To roll back the predicate, manually re-run d4e9a1c7b520's
    `op.execute(...)` against the DB."""
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
