"""seed the ruler reference and make ruler frames measurable

Revision ID: b4c81f60d7e2
Revises: a91d4e70c3b8
Create Date: 2026-08-04 21:00:00.000000

Every head/tail label on a ruler frame currently marks the same 14-inch span,
so the ruler is a fixed-length reference like the fish models and measures
through the same name-keyed path.

It is worth more than a second data point: a fish model's tail landmark is
ambiguous (fork vs tip, worth ~5pp), while a ruler's endpoints are not. Grading
the ruler therefore separates calibration error from labeling convention, which
no model can do.

Recreates `dive_pipeline_status` because the measurable predicate widened; the
accuracy view is untouched (it joins `fishmodelreference` on `Fish.name`, so
"Ruler" flows through unchanged once the reference row exists).

"""
# pylint: skip-file

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from fishsense_api.views import (
    DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
    KNOWN_FISH_MODELS,
)


# revision identifiers, used by Alembic.
revision: str = "b4c81f60d7e2"
down_revision: Union[str, Sequence[str], None] = "a91d4e70c3b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Seed the ruler reference row, then widen the measurable predicate."""
    bind = op.get_bind()
    existing = {
        row[0]
        for row in bind.execute(sa.text("SELECT name FROM fishmodelreference"))
    }
    to_insert = [m for m in KNOWN_FISH_MODELS if m["name"] not in existing]
    if to_insert:
        bind.execute(
            sa.text(
                "INSERT INTO fishmodelreference (name, known_length_m) "
                "VALUES (:name, :known_length_m)"
            ),
            to_insert,
        )

    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)


def downgrade() -> None:
    """Narrow the predicate back. The seeded row is left in place — it is inert
    without the predicate and dropping it would discard a hand-corrected span.
    """
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)
