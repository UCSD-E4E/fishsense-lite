"""add headtailprediction table

Revision ID: e9c1a7b40d53
Revises: d81b6c4a5f27
Create Date: 2026-09-03 00:00:00.000000

The store for model-assisted head/tail labeling: a SAM3 mask of the fish the
validated laser dot sits on, keypointed by `fishsense_core`'s
`FishHeadTailDetector`, emitted by the populate step as a Label Studio
pre-annotation.

Its own table rather than columns on `headtaillabel`, for the same reason
`laserprediction` is separate from `laserlabel`: a prediction must never count
toward a human "valid head/tail" gate, and every one of those gates keys on the
existence of a label row. A labeler's confirmation still lands as an ordinary
`HeadTailLabel` through the usual sync.

Nothing to backfill. An absent row reads as "not predicted yet", which is what
the cohort selects on.

`predictor_version` is what makes improving this stage a drainable cohort
rather than a hand-run backfill — the cohort selects on a *mismatch* with the
current version. It matters more here than it did for the laser stage, because
two of this stage's inputs are invisible to any checkpoint hash: the mask
backend (fishsense-core Mask R-CNN -> SAM3) and the crop window, which is a
*tuned* parameter. Nullable, because "unknown, therefore stale" is the honest
reading for any row written before a bump.

`status` and `rejected_low_confidence` both carry a `server_default` and NOT
NULL, the c7e4a91f2d38 lesson: a NULL backfill makes `WHERE NOT
rejected_low_confidence` silently drop rows under three-valued logic, and a
NULL `status` would make an abstention indistinguishable from a prediction.

The keypoint columns are genuinely nullable — all four are NULL on an
abstention, and `status` says which kind of abstention it was.
"""

# pylint: skip-file

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision: str = "e9c1a7b40d53"
down_revision: Union[str, Sequence[str], None] = "d81b6c4a5f27"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Idempotent against `SQLModel.metadata.create_all`: the FastAPI lifespan
    runs `create_all` *before* the alembic upgrade, and `headtailprediction` is
    in the ORM model registry, so on an existing DB the table already exists by
    the time this migration runs. Skip the DDL when it is present rather than
    crash startup with `DuplicateTableError` — that is the failure that took
    fishsense-api down once already.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("headtailprediction"):
        return
    op.create_table(
        "headtailprediction",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("head_x", sa.Float(), nullable=True),
        sa.Column("head_y", sa.Float(), nullable=True),
        sa.Column("tail_x", sa.Float(), nullable=True),
        sa.Column("tail_y", sa.Float(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("mask_area_px", sa.Integer(), nullable=True),
        sa.Column("silhouette_ratio", sa.Float(), nullable=True),
        sa.Column("crop_x", sa.Integer(), nullable=True),
        sa.Column("crop_y", sa.Integer(), nullable=True),
        sa.Column("laser_label_id", sa.Integer(), nullable=True),
        sa.Column("predictor_version", sa.Integer(), nullable=True),
        sa.Column("checkpoint", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("core_version", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column(
            "status",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="predicted",
        ),
        sa.Column(
            "rejected_low_confidence",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("image_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["image_id"], ["image.id"]),
        sa.ForeignKeyConstraint(["laser_label_id"], ["laserlabel.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("image_id", name="uq_headtail_prediction_image"),
    )
    op.create_index(
        op.f("ix_headtailprediction_predictor_version"),
        "headtailprediction",
        ["predictor_version"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_headtailprediction_predictor_version"),
        table_name="headtailprediction",
    )
    op.drop_table("headtailprediction")
