"""add fish_length_estimate view (p90 over a fish's frames)

Revision ID: d7e21b95c3a0
Revises: c5d93a17e8b4
Create Date: 2026-08-04 23:30:00.000000

Exposes the length estimate that should actually be USED for a fish, which is
the p90 over its frames rather than the mean.

Stage 14 back-projects head and tail at a single laser-derived depth, so it
measures the fish's projection: out-of-plane angle can only ever SHORTEN it.
Per-frame error is one-sided-negative (skew -4.87 across 437 fish-model
frames), so the mean is biased low and a high quantile is the right estimator.
Measured over 23 dive x model groups, mean absolute error 4.35% vs p90 2.26% —
a halving, for free, with no change to any measurement.

Additive: nothing reads this yet, and no existing view or query changes.

"""

# pylint: skip-file

from typing import Sequence, Union

from alembic import op

from fishsense_api.views import (
    DROP_FISH_LENGTH_ESTIMATE_VIEW_SQL,
    FISH_LENGTH_ESTIMATE_VIEW_SQL,
)

# revision identifiers, used by Alembic.
revision: str = "d7e21b95c3a0"
down_revision: Union[str, Sequence[str], None] = "c5d93a17e8b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the view (drop first so re-running is safe)."""
    op.execute(DROP_FISH_LENGTH_ESTIMATE_VIEW_SQL)
    op.execute(FISH_LENGTH_ESTIMATE_VIEW_SQL)


def downgrade() -> None:
    """Drop the view."""
    op.execute(DROP_FISH_LENGTH_ESTIMATE_VIEW_SQL)
