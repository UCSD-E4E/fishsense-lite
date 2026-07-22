"""`measured` ignores species rows with no scientific name

`measured`'s "measurable image" predicate checked top-three + valid laser
+ valid headtail + LABEL_STUDIO cluster, but never looked at what the
species label actually said.

Stage 14 does. `measure_fish_activity._parse_species_names` takes the last
", "-separated chunk of `content_of_image` and requires the
`Common Name (Scientific name)` shape, skipping the image otherwise rather
than writing a malformed Species row. Only the `Fish` taxonomy branch has
that shape:

    "Fish, Hogfish (Lachnolaimus maximus)"  -> measurable
    "Fish Model, Weasly Fish"               -> skipped (no parens)
    "Calibration Targets, Ruler"            -> skipped (no parens)

So the view (and the cohort selector that mirrors it) counted images the
activity can never measure. That disagreement cannot resolve on its own:
no Measurement is ever written for such an image, so it stays "measurable
and unmeasured" forever — `measured` is pinned false and the stage-14
cohort re-selects the dive every hour. It is the same never-goes-false
shape that b7c2e4d81a09 fixed for unbound clusters, one layer down.

Latent rather than firing when written: the measure-fish cohort was empty
in prod (`select-next/measure-fish` -> null), because no FishModels dive
had LABEL_STUDIO clusters yet. It would have started firing as soon as one
did — which is exactly the direction those dives are headed, since the
`Fish Model` and `Calibration Targets` branches only ever appear on them.

Drop + recreate rather than CREATE OR REPLACE — postgres is restrictive
about column-shape changes and the view has no dependents.

Revision ID: c8d3f5a21b74
Revises: b7c2e4d81a09
Create Date: 2026-07-21 21:05:00.000000

"""
# pylint: skip-file

from typing import Sequence, Union

from alembic import op

from fishsense_api.views import (
    DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
)

# revision identifiers, used by Alembic.
revision: str = "c8d3f5a21b74"
down_revision: Union[str, Sequence[str], None] = "b7c2e4d81a09"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)


def downgrade() -> None:
    """Drop the new view; the prior migration's upgrade is the recreate.
    To roll back the predicate, manually re-run b7c2e4d81a09's
    `op.execute(...)` against the DB."""
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
