"""add dive_name to dive_pipeline_status view

Superset's pipeline-status dashboard reads this view keyed by `dive_id`,
which is opaque to a human scanning the board. Expose `d.name AS dive_name`
so the dashboard can show the readable dive name (e.g.
`083023_FishModels_FSL05`) alongside the stage flags.

Drop + recreate rather than CREATE OR REPLACE — postgres is restrictive
about column-shape changes and the view has no dependents.

Revision ID: f4a9c2b17e60
Revises: b8e3f1a09d24
Create Date: 2026-07-28 00:00:00.000000

"""

# pylint: skip-file

from typing import Sequence, Union

from alembic import op

from fishsense_api.views import (
    DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
)

# revision identifiers, used by Alembic.
revision: str = "f4a9c2b17e60"
down_revision: Union[str, Sequence[str], None] = "b8e3f1a09d24"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)


def downgrade() -> None:
    """Drop the new view; to restore the prior shape, re-run the previous
    view-defining migration's `op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)`
    against a checkout that predates the `dive_name` column."""
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
