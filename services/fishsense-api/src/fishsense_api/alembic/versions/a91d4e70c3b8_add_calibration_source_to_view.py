"""add calibration_source to dive_pipeline_status

Revision ID: a91d4e70c3b8
Revises: f3b81c6d92a4
Create Date: 2026-08-04 19:30:00.000000

`calibrated` says a dive HAS extrinsics; `calibration_source` says whether they
describe this dive's own rig deployment ('own') or were borrowed from another
dive ('borrowed'). Measured against the known-length fish models, that
distinction is worth ~1% vs -8..+2% accuracy, so downstream analysis needs it.

Drop + recreate rather than CREATE OR REPLACE — Postgres is restrictive about
column-shape changes and the view has no dependents.

"""

# pylint: skip-file

from typing import Sequence, Union

from alembic import op

from fishsense_api.views import (
    DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
)

# revision identifiers, used by Alembic.
revision: str = "a91d4e70c3b8"
down_revision: Union[str, Sequence[str], None] = "f3b81c6d92a4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Recreate the view with the calibration_source column."""
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)


def downgrade() -> None:
    """Recreate from whatever views.py currently defines."""
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)
