"""recreate dive_pipeline_status view for species cascade

Revision ID: 8c8bbdb06f38
Revises: 60e82ad5dac7
Create Date: 2026-05-05

`dive_images_preprocessed` predicate flipped: now requires that every
*laser-valid* image (completed, not superseded, x/y both set) carries
a non-sentinel SpeciesLabel row, instead of every image. Mirrors the
2026-05-05 cohort flip on the species preprocessing pipeline (see
`select_next_for_species_preprocessing`).

CREATE OR REPLACE VIEW is restrictive about column-shape changes in
Postgres; the drop+recreate pattern is simpler and the view has no
dependents in this schema.
"""
# pylint: skip-file

from typing import Sequence, Union

from alembic import op

from fishsense_api.views import (
    DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
)


# revision identifiers, used by Alembic.
revision: str = '8c8bbdb06f38'
down_revision: Union[str, Sequence[str], None] = '60e82ad5dac7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)


def downgrade() -> None:
    """Drop the new view; the prior migration's upgrade is the
    recreate. To roll back the predicate, manually re-run the prior
    migration's `op.execute(...)` against the DB."""
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
