"""add the auto-accept gate verdict to laserprediction

The data-worker's `laser_label_validation.auto_accept` gate decides, per dive,
which laser predictions agree with the dive's own fitted laser line well enough
to skip human review; the api-worker's laser populate step consumes the answer.
These four columns carry it between the two.

`auto_accept` is NOT NULL with `server_default=false`, and that direction is
the safety property: every row predating the gate reads "not auto-acceptable",
i.e. send it to a person. A nullable column would make `WHERE NOT auto_accept`
skip every one of them under three-valued logic — the same trap
`rejected_out_of_region` and `needs_reprocess` were shaped to avoid, but with a
worse consequence, because here the predicate decides whether a human ever
looks at the frame.

The other three stay nullable: NULL means "the gate never ran on this row",
which must stay distinguishable from a gate that ran and declined.

Revision ID: c4f8a2e60b17
Revises: e9c1a7b40d53
Create Date: 2026-09-03 00:00:00.000000

"""

# pylint: skip-file

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4f8a2e60b17"
down_revision: Union[str, Sequence[str], None] = "e9c1a7b40d53"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_COLUMNS = (
    ("auto_accept", sa.Boolean(), False, sa.false()),
    ("gate_verdict", sa.String(), True, None),
    ("line_offset_px", sa.Float(), True, None),
    ("line_position_z", sa.Float(), True, None),
)


def upgrade() -> None:
    """Upgrade schema.

    Idempotent against `SQLModel.metadata.create_all`: `lifespan` runs
    `create_all` before alembic, so on a fresh database these columns already
    exist by the time this runs and a bare `add_column` would raise
    `DuplicateColumnError` and stop the API from starting.
    """
    bind = op.get_bind()
    existing = {c["name"] for c in sa.inspect(bind).get_columns("laserprediction")}
    for name, type_, nullable, default in _COLUMNS:
        if name in existing:
            continue
        op.add_column(
            "laserprediction",
            sa.Column(name, type_, nullable=nullable, server_default=default),
        )
    indexes = {i["name"] for i in sa.inspect(bind).get_indexes("laserprediction")}
    if "ix_laserprediction_gate_verdict" not in indexes:
        op.create_index(
            "ix_laserprediction_gate_verdict",
            "laserprediction",
            ["gate_verdict"],
        )


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    indexes = {i["name"] for i in sa.inspect(bind).get_indexes("laserprediction")}
    if "ix_laserprediction_gate_verdict" in indexes:
        op.drop_index("ix_laserprediction_gate_verdict", "laserprediction")
    for name, _, _, _ in reversed(_COLUMNS):
        op.drop_column("laserprediction", name)
