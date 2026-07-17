"""rescope `measured` in dive_pipeline_status to measurable images

`measured` was "≥1 LABEL_STUDIO cluster AND zero with fish_id NULL",
which is unreachable. A cluster is only bound to a fish through a
top-three species label whose image has a valid laser + headtail, so any
LABEL_STUDIO cluster without such an image held the dive at
measured=false permanently.

That is not theoretical: at the time of writing every one of the 8
calibrated dives in prod was pinned false, and dive 466 carried 1632
unbound clusters against only 24 measurable images (the residue of
repeated stage-6.1 POSTs in the notebook era — the cluster API has no
DELETE). The stage-14 cohort selector mirrors this predicate, so a
scheduled stage 14 would have re-selected the same dives every hour
forever.

`measured` now means: ≥1 measurement for the dive AND no measurable
image left unmeasured, where "measurable" mirrors what
measure_fish_activity actually attempts. Validated against prod before
writing: 7 of the 8 dives flip to true, and 466 reads false because it
genuinely has 1 measurable image left.

Drop + recreate rather than CREATE OR REPLACE — postgres is restrictive
about column-shape changes and the view has no dependents.

Revision ID: b7c2e4d81a09
Revises: a3f1c7d94b52
Create Date: 2026-07-17 03:55:00.000000

"""
# pylint: skip-file

from typing import Sequence, Union

from alembic import op

from fishsense_api.views import (
    DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
)

# revision identifiers, used by Alembic.
revision: str = "b7c2e4d81a09"
down_revision: Union[str, Sequence[str], None] = "a3f1c7d94b52"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)


def downgrade() -> None:
    """Drop the new view; the prior migration's upgrade is the
    recreate. To roll back the predicate, manually re-run the prior
    migration's `op.execute(...)` against the DB."""
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
