"""make fish models measurable in dive_pipeline_status

Revision ID: d8b3f16c204e
Revises: c7f2a9d14b83
Create Date: 2026-08-04 00:10:00.000000

`dive_pipeline_status.measured` now counts physical fish-model images
(`content_of_image LIKE 'Fish Model,%'`) as measurable, and waives the
LABEL_STUDIO-cluster requirement for them (models carry no grouping labels and
so no cluster). Canonical SQL lives in `views.py`; drop + recreate rather than
CREATE OR REPLACE — Postgres is restrictive about it and the view has no
dependents.

"""
# pylint: skip-file

from typing import Sequence, Union

from alembic import op

from fishsense_api.views import (
    DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
)


# revision identifiers, used by Alembic.
revision: str = "d8b3f16c204e"
down_revision: Union[str, Sequence[str], None] = "c7f2a9d14b83"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Recreate the view with the fish-model-aware `measured` predicate."""
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)


def downgrade() -> None:
    """Recreate the view from whatever `views.py` currently defines."""
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)
