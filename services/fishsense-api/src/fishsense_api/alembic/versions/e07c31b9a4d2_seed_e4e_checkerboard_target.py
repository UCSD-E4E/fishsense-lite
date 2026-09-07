"""seed the E4E checkerboard calibration target

`d92a1f4c78b3` created `calibrationtarget` and deliberately seeded nothing: the
table is useless without a measured `square_size_m`, and that number is a
caliper reading of a physical board, not something a migration can know. This
is that reading arriving — 4.2 cm, reported 2026-09-07. Provenance and its
uncertainty live in `views.KNOWN_CALIBRATION_TARGETS`, on the row's `notes`.

Until this ran, the whole checkerboard calibration path was complete and
calibrated nothing: the cohort selects on `Dive.calibration_target_id`, and
species sync can only set that to a row that exists.

**Insert-only.** An operator who re-measures the board across many squares —
which is the standing advice, and is worth ~10x on this number — must not have
their correction stamped back on the next deploy. Same posture as the
`fishmodelreference` seed, and for the same reason.

Revision ID: e07c31b9a4d2
Revises: d92a1f4c78b3
Create Date: 2026-09-07 00:00:00.000000

"""

# pylint: skip-file

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from fishsense_api.views import KNOWN_CALIBRATION_TARGETS

# revision identifiers, used by Alembic.
revision: str = "e07c31b9a4d2"
down_revision: Union[str, Sequence[str], None] = "d92a1f4c78b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Insert any missing calibration target, leaving existing rows alone."""
    bind = op.get_bind()
    present = {
        row[0] for row in bind.execute(sa.text("SELECT name FROM calibrationtarget"))
    }
    # Named keys rather than passing the dicts through: adding a key to
    # `KNOWN_CALIBRATION_TARGETS` later must not retroactively change what this
    # already-applied migration did.
    for target in KNOWN_CALIBRATION_TARGETS:
        if target["name"] in present:
            continue
        bind.execute(
            sa.text(
                "INSERT INTO calibrationtarget "
                "(name, rows, cols, square_size_m, notes) "
                "VALUES (:name, :rows, :cols, :square_size_m, :notes)"
            ),
            {
                "name": target["name"],
                "rows": target["rows"],
                "cols": target["cols"],
                "square_size_m": target["square_size_m"],
                "notes": target.get("notes"),
            },
        )


def downgrade() -> None:
    """Remove only the rows this migration seeds.

    By name, so a target an operator added by hand survives — and so does a
    seeded row they have since re-measured, which is a judgement call this
    downgrade is not entitled to make. It removes the name; the operator's
    correction to that name goes with it, which is the point of a downgrade.
    """
    bind = op.get_bind()
    for target in KNOWN_CALIBRATION_TARGETS:
        bind.execute(
            sa.text("DELETE FROM calibrationtarget WHERE name = :name"),
            {"name": target["name"]},
        )
