"""at most one canonical image per checksum

Revision ID: a7f2c9e41b03
Revises: d7e21b95c3a0
Create Date: 2026-08-07 09:00:00.000000

`post_image` decides `is_canonical` with a read-then-write, which under READ
COMMITTED lets two concurrent requests both conclude they are the first row for
a checksum. Only the database can settle that, so this adds the partial unique
index the application logic has always assumed.

**This migration never mutates data.** If existing rows already violate the
invariant, creating the index would fail -- and because `lifespan` runs
`run_alembic_upgrade` on startup, a failed migration crash-loops the API. So it
checks first and, on finding violations, logs them loudly and skips. Repairing
them is an operator decision (which copy is canonical is a judgement about
which dive is the real one), not something a schema migration should decide
unattended against a database with no staging counterpart.

To repair, demote all but the earliest row per checksum -- `9e5bc64`'s original
rule was "first one wins" -- then re-run this migration:

    UPDATE image SET is_canonical = false
    WHERE is_canonical AND id NOT IN (
        SELECT MIN(id) FROM image WHERE is_canonical GROUP BY checksum
    );

"""

# pylint: skip-file

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7f2c9e41b03"
down_revision: Union[str, Sequence[str], None] = "d7e21b95c3a0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_log = logging.getLogger("alembic.runtime.migration")

INDEX_NAME = "uq_image_canonical_checksum"

_FIND_VIOLATIONS = sa.text("""
    SELECT checksum, COUNT(*) AS n
    FROM image
    WHERE is_canonical
    GROUP BY checksum
    HAVING COUNT(*) > 1
    ORDER BY n DESC
    """)


def upgrade() -> None:
    bind = op.get_bind()

    if INDEX_NAME in {ix["name"] for ix in sa.inspect(bind).get_indexes("image")}:
        _log.info("%s already present; nothing to do", INDEX_NAME)
        return

    violations = list(bind.execute(_FIND_VIOLATIONS))
    if violations:
        total = sum(row.n for row in violations)
        _log.error(
            "REFUSING to create %s: %d checksums already have more than one "
            "canonical image (%d rows). Nothing has been changed and the "
            "upgrade continues, so the API still starts -- but the race this "
            "index closes remains open. See this migration's docstring for the "
            "repair statement. Worst offenders: %s",
            INDEX_NAME,
            len(violations),
            total,
            ", ".join(f"{row.checksum}={row.n}" for row in violations[:5]),
        )
        return

    op.create_index(
        INDEX_NAME,
        "image",
        ["checksum"],
        unique=True,
        postgresql_where=sa.text("is_canonical"),
        sqlite_where=sa.text("is_canonical"),
    )
    _log.info("created %s", INDEX_NAME)


def downgrade() -> None:
    bind = op.get_bind()
    if INDEX_NAME in {ix["name"] for ix in sa.inspect(bind).get_indexes("image")}:
        op.drop_index(INDEX_NAME, table_name="image")
