"""add dive_pipeline_status view

Revision ID: 60e82ad5dac7
Revises: 299428e39cb9
Create Date: 2026-05-04

"""
# pylint: skip-file

from typing import Sequence, Union

from alembic import op

from fishsense_api.views import (
    DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
)


# revision identifiers, used by Alembic.
revision: str = '60e82ad5dac7'
down_revision: Union[str, Sequence[str], None] = '299428e39cb9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the read-only dive_pipeline_status view.

    Backs Superset dashboards against the fishsense Postgres DB. The
    view derives one boolean column per pipeline stage from the
    underlying tables — no writeback path, no drift. SQL lives in
    `fishsense_api.views` so this migration and the test fixture
    apply the same definition.
    """
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)


def downgrade() -> None:
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
