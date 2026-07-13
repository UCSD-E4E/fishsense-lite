"""add superseded to species + dive_slate labels

Revision ID: 7934e62a12c0
Revises: 8c8bbdb06f38
Create Date: 2026-07-13

Gives SpeciesLabel + DiveSlateLabel the `superseded` dead-letter flag that
laser + headtail already carry, so all four label types have uniform
dead-letter semantics.

Backfills existing rows to FALSE: the view/query predicates use
`superseded = FALSE`, and `NULL = FALSE` is NULL (not true), so a column
added nullable-without-backfill would silently drop every existing
species/slate row out of the `*_labeling_complete` / read predicates.

Then drops + recreates `dive_pipeline_status` so its
`species_labeling_complete` / `slate_labeling_complete` columns pick up the
new `superseded = FALSE` predicate (view SQL is canonicalized in
fishsense_api.views).
"""
# pylint: skip-file

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from fishsense_api.views import (
    DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
)


# revision identifiers, used by Alembic.
revision: str = "7934e62a12c0"
down_revision: Union[str, Sequence[str], None] = "8c8bbdb06f38"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("specieslabel", sa.Column("superseded", sa.Boolean(), nullable=True))
    op.execute("UPDATE specieslabel SET superseded = FALSE WHERE superseded IS NULL")
    op.add_column("diveslatelabel", sa.Column("superseded", sa.Boolean(), nullable=True))
    op.execute("UPDATE diveslatelabel SET superseded = FALSE WHERE superseded IS NULL")

    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)


def downgrade() -> None:
    """Drop the columns; the view is left dropped rather than recreated —
    `DIVE_PIPELINE_STATUS_VIEW_SQL` now references `superseded`, so it can't
    be rebuilt after the columns are gone. To restore the old view, re-run
    the prior migration's upgrade from a checkout that predates this one
    (same convention as 8c8bbdb06f38's downgrade)."""
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.drop_column("diveslatelabel", "superseded")
    op.drop_column("specieslabel", "superseded")
