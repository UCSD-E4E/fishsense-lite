"""correct the ruler reference from 14 in to the labeled 13.5 in span

Revision ID: c5d93a17e8b4
Revises: b4c81f60d7e2
Create Date: 2026-08-04 22:15:00.000000

b4c81f60d7e2 seeded the ruler at 0.3556 m (14 in, its nominal size). That is the
wrong reference: labelers do not click a physical zero, they click the first
*printed* graduation — the 0.5 mark, the leftmost thing on the scale — and the
14 mark. The labeled span is therefore 0.5->14 = 13.5 in = 0.3429 m.

Measured off the ruler's own inch ticks on four frames: 13.500 / 13.505 /
13.481 / 13.468 in (SD 0.13%), corroborated by a second, independent method
(half-inch ticks fitted with a 1D projective map, which also removes
perspective). The 14 in figure survived a photo check only because the photo
confirmed the TAIL end; nobody had looked at the head end.

No re-measurement is needed — `fish_model_measurement_accuracy` derives
pct_error from this row at query time, so the existing ruler measurements
re-grade themselves (roughly -5.4% -> -2.4% on the best frames).

"""

# pylint: skip-file

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "c5d93a17e8b4"
down_revision: Union[str, Sequence[str], None] = "b4c81f60d7e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_OLD = 0.3556
_NEW = 0.3429


def upgrade() -> None:
    """Point the ruler reference at the labeled span.

    Guarded on the old value so this never clobbers a length an operator has
    since corrected by hand — the same posture as the original seed, which only
    inserted names that were absent.
    """
    op.get_bind().execute(
        sa.text(
            "UPDATE fishmodelreference SET known_length_m = :new "
            "WHERE name = 'Ruler' AND abs(known_length_m - :old) < 1e-6"
        ),
        {"new": _NEW, "old": _OLD},
    )


def downgrade() -> None:
    """Restore the nominal 14 in value."""
    op.get_bind().execute(
        sa.text(
            "UPDATE fishmodelreference SET known_length_m = :old "
            "WHERE name = 'Ruler' AND abs(known_length_m - :new) < 1e-6"
        ),
        {"new": _NEW, "old": _OLD},
    )
