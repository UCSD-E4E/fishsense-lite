"""Seed the `Weasly Fish` reference row so its measurements can be graded.

`Weasly Fish` has been a pickable `Fish Model` leaf in the species labeling XML
all along, with no `fishmodelreference` row. Because
`fish_model_measurement_accuracy` INNER JOINs measurements to the reference on
`Fish.name`, the consequence was silence rather than an error: every Weasly Fish
measurement was absent from the accuracy view, not wrong in it. Nothing counted
wrong, nothing logged, and the coverage test passed because it only checked four
names someone had already thought of.

The length is PROVISIONAL (0.30 m, an estimate). The body widths are not — they
are calipered, and are recorded in `notes` as an independent input for the
round-model thickness work. Thickness must never be back-solved from measurement
error: doing so absorbs whatever calibration bias is present and turns the
held-out validation set into something that grades itself.

Seed-only-if-absent, like every other reference seed here: an operator may have
corrected a length by hand, and a migration must not stamp over that.

Revision ID: d1a7b3e95c02
Revises: e3f70ac46d92
"""

# pylint: skip-file
# Convention across all 71 other migrations here: alembic's generated module
# shape (lowercase `revision` / `down_revision` module constants, `op.get_bind`
# resolved dynamically) trips invalid-name and no-member with nothing useful to
# say about it.

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from fishsense_api.views import FISH_MODEL_NOTES, KNOWN_FISH_MODELS

revision: str = "d1a7b3e95c02"
down_revision: Union[str, Sequence[str], None] = "e3f70ac46d92"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NAME = "Weasly Fish"


def upgrade() -> None:
    """Insert the reference row if it isn't already there."""
    bind = op.get_bind()

    existing = {
        row[0] for row in bind.execute(sa.text("SELECT name FROM fishmodelreference"))
    }
    if _NAME in existing:
        return

    model = next(m for m in KNOWN_FISH_MODELS if m["name"] == _NAME)
    bind.execute(
        sa.text(
            "INSERT INTO fishmodelreference (name, known_length_m, notes) "
            "VALUES (:name, :known_length_m, :notes)"
        ),
        {
            "name": model["name"],
            "known_length_m": model["known_length_m"],
            "notes": FISH_MODEL_NOTES[_NAME],
        },
    )


def downgrade() -> None:
    """Remove the seeded row.

    Safe because the row is pure reference data — no measurement points at it;
    the accuracy view joins BY NAME, so dropping it returns Weasly Fish
    measurements to being ungradeable rather than orphaning anything.
    """
    op.get_bind().execute(
        sa.text("DELETE FROM fishmodelreference WHERE name = :name"),
        {"name": _NAME},
    )
