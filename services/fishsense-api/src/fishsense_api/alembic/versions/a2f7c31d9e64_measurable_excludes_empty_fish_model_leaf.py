"""dive_pipeline_status: measurable excludes the empty "Fish Model," leaf

Revision ID: a2f7c31d9e64
Revises: c3e8b1d47a52
Create Date: 2026-08-11 00:00:00.000000

The measurable-species predicate matched `content_of_image = "Fish Model,"` —
the parent taxonomy node with no model leaf, which a labeler produces by
selecting "Fish Model" and not picking one. `LIKE 'Fish Model,%'` matches it
because `%` matches the empty string.

`measure_fish_activity` does NOT accept it: `parse_model_name` returns None for
an empty (or whitespace-only) leaf, so the activity skips the image rather than
writing a Fish with an empty `name`. Cohort says measurable, activity says
skip — no Measurement is ever written, `NOT EXISTS (measurement)` stays true,
and the dive is re-selected every hour forever. That is the identical
never-goes-false shape that blocked scheduling stage 14 before 2026-07-17,
reachable from a single labeler mis-click.

The predicate now adds `AND TRIM(content_of_image) <> 'Fish Model,'`, which
also covers a whitespace-only leaf (matching `parse_model_name`'s `.strip()`).
`TRIM` behaves identically on Postgres (prod) and SQLite (tests).

Found by `test_measurable_species_sql_agrees_with_taxonomy_is_measurable`,
which runs this SQL and `fishsense_shared.taxonomy.is_measurable` over the
shared `MEASURABILITY_CORPUS` and asserts they select the same rows. Before
this revision the three copies of the predicate were kept in step by comments
only — comments that had themselves drifted into claiming fish models and the
ruler were unmeasurable.

Drop + recreate rather than CREATE OR REPLACE: Postgres is restrictive about
column-shape changes on replace, and the view has no dependents (see the
`dive_pipeline_status` section of CLAUDE.md).

No data is touched, and the column shape is unchanged. On prod the effect is
confined to dives holding at least one empty-leaf `"Fish Model,"` species row:
those stop counting toward `measured`'s denominator, so a dive wedged on one
can finally read `measured = true` and drain from the stage-14 cohort. Every
other dive's flags are byte-identical.
"""

# pylint: skip-file

from typing import Sequence, Union

from alembic import op

from fishsense_api.views import (
    DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
)

# revision identifiers, used by Alembic.
revision: str = "a2f7c31d9e64"
down_revision: Union[str, Sequence[str], None] = "c3e8b1d47a52"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Recreate the view with the empty-leaf guard."""
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)


def downgrade() -> None:
    """Recreate from whatever `views.py` currently holds.

    Not a true inverse: the SQL is imported from `views.py`, the single source
    of truth, so downgrading after reverting the code restores the old
    predicate while downgrading without reverting is a no-op. Same trade-off
    every view migration in this tree makes.
    """
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)
