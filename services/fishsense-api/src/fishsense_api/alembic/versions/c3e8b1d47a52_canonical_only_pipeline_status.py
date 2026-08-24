"""dive_pipeline_status counts only canonical images

Revision ID: c3e8b1d47a52
Revises: a7f2c9e41b03
Create Date: 2026-08-08 06:00:00.000000

The cohort selectors now require `Image.is_canonical`, so a duplicate dive can
never become pipeline work. This view has to use the identical predicate or the
two drift: the dashboard would report such a dive as perpetually incomplete
while the worker correctly does nothing with it — the exact silent
disagreement `test_view_and_selector_agree_on_species_predicate` exists to
catch.

Drop + recreate rather than CREATE OR REPLACE: Postgres is restrictive about
column-shape changes on replace, and the view has no dependents, so the simpler
pattern applies (see the `dive_pipeline_status` section of CLAUDE.md).

No data is touched. On prod this is a pure predicate change: every dive that
currently has canonical images keeps identical flags, because the added
conjunct is true for those rows. Only fully-duplicate dives change — and all
207 of them are priority=LOW, so nothing schedules them either way.

"""

# pylint: skip-file

from typing import Sequence, Union

from alembic import op

from fishsense_api.views import (
    DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
)

# revision identifiers, used by Alembic.
revision: str = "c3e8b1d47a52"
down_revision: Union[str, Sequence[str], None] = "a7f2c9e41b03"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Recreate the view with the canonical-only predicate."""
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)


def downgrade() -> None:
    """Recreate from whatever `views.py` currently holds.

    Note this is NOT a true inverse: the SQL is imported from `views.py`, which
    is the single source of truth, so downgrading after reverting the code
    restores the old predicate, while downgrading without reverting is a no-op.
    That is the same trade-off every view migration in this tree makes.
    """
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)
