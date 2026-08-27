"""add Dive.notes and Priority.NONE

Revision ID: b3d5e91a7c42
Revises: a4c17f2b9e08
Create Date: 2026-08-27 00:00:00.000000

`NONE` is a third priority meaning "deliberately excluded", as opposed to
LOW's "not yet". `notes` carries the reason. See `models/priority.py`.

Two dialect-specific details, both of which fail loudly if got wrong:

* `priority` is a Postgres *native enum type*, so a new member needs
  `ALTER TYPE ... ADD VALUE`. SQLite renders the same column as VARCHAR and
  has no such statement, so the call is guarded on the dialect — the
  migration tests in this repo run against in-memory SQLite.
* `IF NOT EXISTS` is required, not belt-and-braces. `lifespan` runs
  `SQLModel.metadata.create_all` before `run_alembic_upgrade`, so a fresh
  database already has the type built from the model *including* `NONE`
  by the time this runs.

`ALTER TYPE ... ADD VALUE` has been transaction-safe since Postgres 12
(prod is 17), so it needs no autocommit block. The added label must not be
*used* in the same transaction, and nothing here does.
"""

# pylint: skip-file

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b3d5e91a7c42"
down_revision: Union[str, Sequence[str], None] = "a4c17f2b9e08"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

ENUM_TYPE_NAME = "priority"
NEW_MEMBER = "NONE"


def add_enum_value_sql(dialect_name: str) -> str | None:
    """The `ALTER TYPE` for `dialect_name`, or None where it doesn't apply.

    Split out as a plain function so the Postgres-only statement is
    assertable from a SQLite test run.
    """
    if dialect_name != "postgresql":
        return None
    return f"ALTER TYPE {ENUM_TYPE_NAME} ADD VALUE IF NOT EXISTS '{NEW_MEMBER}'"


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("dive", sa.Column("notes", sa.String(), nullable=True))

    sql = add_enum_value_sql(op.get_bind().dialect.name)
    if sql is not None:
        op.execute(sql)


def downgrade() -> None:
    """Downgrade schema.

    Only `notes` is reversible. Postgres cannot drop a value from an enum
    type, and faking it (rebuild the type, rewrite the column) would have to
    decide what to do with any row already sitting at NONE — silently
    rewriting an operator's deliberate exclusion back to LOW is worse than
    leaving an unused label in the type.
    """
    op.drop_column("dive", "notes")
