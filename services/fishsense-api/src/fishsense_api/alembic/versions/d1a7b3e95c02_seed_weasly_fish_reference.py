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

    # A filtering comprehension, not `next(...)`: this migration is shipped and
    # replays on every fresh database forever. If the model is ever renamed in
    # `KNOWN_FISH_MODELS`, a bare `next` raises StopIteration inside the FastAPI
    # lifespan and the API fails to boot. Degrading to a no-op is the only safe
    # behaviour for a historical seed. Same reason `FISH_MODEL_NOTES` is read
    # with `.get`.
    candidates = [m for m in KNOWN_FISH_MODELS if m["name"] == _NAME]
    if not candidates:
        return
    model = candidates[0]
    bind.execute(
        sa.text(
            "INSERT INTO fishmodelreference (name, known_length_m, notes) "
            "VALUES (:name, :known_length_m, :notes)"
        ),
        {
            "name": model["name"],
            "known_length_m": model["known_length_m"],
            "notes": FISH_MODEL_NOTES.get(_NAME),
        },
    )


def downgrade() -> None:
    """Leave the row in place.

    Deliberately a no-op, matching sibling `b4c81f60d7e2`, which seeds the same
    table and says the same thing. The row is inert without a measurement to
    grade, and `upgrade()` goes out of its way NOT to stamp over a length an
    operator has since calipered — deleting it on the way back down would throw
    away exactly the work that guard exists to protect.
    """
