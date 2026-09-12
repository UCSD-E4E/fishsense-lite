"""Record when a calibration fit was refused, so the dive leaves the cohort.

Both calibration cohorts select on dive *state* -- "has no usable
`LaserExtrinsics` row" -- and a refusal does not change that state. So a dive
whose observations cannot produce a sound fit is re-selected every hour
forever, re-staging its raw `.ORF`s from the NAS each time, and because the
selectors are `ORDER BY id LIMIT 1` it blocks every dive behind it. That is the
prod dive-347 shape, and the baseline gate made it reachable for eight more
dives at once.

Nullable with no default and no backfill, deliberately: absent means "never
refused", which is the correct reading for every existing row.

Revision ID: c4e8b17a205f
Revises: a1d47f2c9b83
"""

# pylint: skip-file
#   `alembic.op` is assembled at runtime via a proxy, so pylint sees no
#   `add_column` / `drop_column` members. Every migration in this directory
#   carries the same pragma.

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c4e8b17a205f"
down_revision: Union[str, Sequence[str], None] = "a1d47f2c9b83"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the refusal timestamp and its reason to `dive`."""
    op.add_column(
        "dive",
        sa.Column("calibration_refused_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "dive",
        sa.Column("calibration_refused_reason", sa.String(), nullable=True),
    )


def downgrade() -> None:
    """Drop both columns.

    Safe in a way the `priority` enum downgrade was not: these are ordinary
    nullable columns, so removing them loses only the recorded refusals and
    returns the cohorts to re-offering those dives — the behaviour before this
    revision.
    """
    op.drop_column("dive", "calibration_refused_reason")
    op.drop_column("dive", "calibration_refused_at")
