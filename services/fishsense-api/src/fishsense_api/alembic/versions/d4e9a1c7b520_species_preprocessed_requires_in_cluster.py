"""`dive_images_preprocessed` counts only in-cluster, live-labeled images

The stage-2 `dive_images_preprocessed` flag checked "dive has a PREDICTION
cluster" and "every valid-laser image has a non-sentinel species row" as two
independent, dive-wide conditions. But `resolve_species_preprocess_inputs_activity`
only makes JPEGs for a qualifying image that is *in* a PREDICTION cluster (it
needs the cluster for the "image i of N" overlay), and
`select_next_for_species_preprocessing` was tightened to match on 2026-07-22.

The view lagged, so it diverged from the selector two ways:

* An image with a valid laser but in no cluster (a laser validated after
  one-shot stage-1 clustering) counted against "preprocessed", leaving the
  dive forever incomplete on the dashboard while the selector — correctly —
  never re-fired on it. In prod this was the poison pill: such a dive sat at
  the front of the one-per-hour selector queue resolving to zero and starving
  every productive dive behind it.
* The species existence check ignored `superseded`, so a dive whose only
  species rows were dead-lettered read as preprocessed=true. The selector
  became superseded-aware in b7c... / the 2026-07-21 preprocess-cohort fix;
  the view did not.

Both are now defined identically to the selector: a "processable" image is
valid-laser AND in a PREDICTION cluster, and only a live (non-superseded)
non-sentinel species row marks it done.

Drop + recreate rather than CREATE OR REPLACE — postgres is restrictive about
column-shape changes and the view has no dependents.

Revision ID: d4e9a1c7b520
Revises: c8d3f5a21b74
Create Date: 2026-07-22 23:55:00.000000

"""
# pylint: skip-file

from typing import Sequence, Union

from alembic import op

from fishsense_api.views import (
    DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
)

# revision identifiers, used by Alembic.
revision: str = "d4e9a1c7b520"
down_revision: Union[str, Sequence[str], None] = "c8d3f5a21b74"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
    op.execute(DIVE_PIPELINE_STATUS_VIEW_SQL)


def downgrade() -> None:
    """Drop the new view; the prior migration's upgrade is the recreate.
    To roll back the predicate, manually re-run c8d3f5a21b74's
    `op.execute(...)` against the DB."""
    op.execute(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL)
