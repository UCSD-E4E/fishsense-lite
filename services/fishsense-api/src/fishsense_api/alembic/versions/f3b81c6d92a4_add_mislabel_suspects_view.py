"""add fish_model_species_mislabel_suspects view

Revision ID: f3b81c6d92a4
Revises: e2c9a4f70b31
Create Date: 2026-08-04 17:00:00.000000

Flags fish-model frames whose measured length fits a different known model
better than their own species label. Foreshortening-aware: see views.py.

"""
# pylint: skip-file

from typing import Sequence, Union

from alembic import op

from fishsense_api.views import (
    DROP_FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL,
    FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL,
)


# revision identifiers, used by Alembic.
revision: str = "f3b81c6d92a4"
down_revision: Union[str, Sequence[str], None] = "e2c9a4f70b31"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the suspects view (drop first so re-runs pick up SQL edits)."""
    op.execute(DROP_FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL)
    op.execute(FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(DROP_FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL)
