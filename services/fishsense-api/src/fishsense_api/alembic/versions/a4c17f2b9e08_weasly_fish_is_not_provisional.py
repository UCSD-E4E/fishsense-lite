"""Clear the provisional flag on Weasly Fish — 31 cm is a measurement.

`d1a7b3e95c02` seeded the row at an eyeball 30 cm and `f2b8c04e71a3` marked it
provisional. Both are superseded: the fork length is 31 cm, the figure used in a
prior publication, known to +-5 mm.

The flag was also doing the wrong job. It excludes a row from
`fish_model_species_mislabel_suspects`' best-fit search, and the reason a model
attracts other models' frames is that it EXISTS at that length, not how the
length was obtained — a foreshortened Grouper lands on a 310 mm reference
whether or not anyone calipered it. The view already accepts that class of
ambiguity for real models (Snook reading 15% short is flagged as Grouper) and
marks it `medium`; singling this row out was inconsistent.

Clears the flag wherever it is set, driven by `PROVISIONAL_FISH_MODELS` (now
empty) rather than a literal, so the two cannot disagree.

Revision ID: a4c17f2b9e08
Revises: f2b8c04e71a3
"""

# pylint: skip-file
# See d1a7b3e95c02 — alembic's generated module shape trips invalid-name and
# no-member across every migration here.

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from fishsense_api.views import PROVISIONAL_FISH_MODELS

revision: str = "a4c17f2b9e08"
down_revision: Union[str, Sequence[str], None] = "f2b8c04e71a3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Set `is_provisional` to match the declared set, in both directions."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("fishmodelreference"):
        return
    if "is_provisional" not in {
        c["name"] for c in inspector.get_columns("fishmodelreference")
    }:
        return

    # Reconcile rather than blanket-clear: a row named in the set stays flagged,
    # so re-adding a genuinely unmeasured model later needs no new migration.
    names = sorted(PROVISIONAL_FISH_MODELS)
    if names:
        bind.execute(
            sa.text(
                "UPDATE fishmodelreference SET is_provisional = TRUE "
                "WHERE name IN :names"
            ).bindparams(sa.bindparam("names", expanding=True)),
            {"names": names},
        )
        bind.execute(
            sa.text(
                "UPDATE fishmodelreference SET is_provisional = FALSE "
                "WHERE name NOT IN :names"
            ).bindparams(sa.bindparam("names", expanding=True)),
            {"names": names},
        )
    else:
        bind.execute(sa.text("UPDATE fishmodelreference SET is_provisional = FALSE"))


def downgrade() -> None:
    """No-op. Re-flagging a measured length would be reintroducing the error."""
