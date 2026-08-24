"""add nullable-unique fish.name

Revision ID: c7f2a9d14b83
Revises: a3d0440cda27
Create Date: 2026-08-04 00:00:00.000000

"""

# pylint: skip-file

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "c7f2a9d14b83"
down_revision: Union[str, Sequence[str], None] = "a3d0440cda27"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add `fish.name` + a nullable-unique constraint.

    Idempotent against `SQLModel.metadata.create_all` (the lifespan runs it
    before alembic upgrade): on a fresh DB the column + constraint already
    exist from the ORM model, so skip; on an existing prod DB `fish` predates
    the column, so add it. Existing rows keep name=NULL, and multiple NULLs are
    allowed under a UNIQUE constraint, so the constraint applies cleanly.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("fish")}
    if "name" not in columns:
        op.add_column("fish", sa.Column("name", sa.String(), nullable=True))
    constraints = {c["name"] for c in inspector.get_unique_constraints("fish")}
    if "uq_fish_name" not in constraints:
        op.create_unique_constraint("uq_fish_name", "fish", ["name"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("uq_fish_name", "fish", type_="unique")
    op.drop_column("fish", "name")
