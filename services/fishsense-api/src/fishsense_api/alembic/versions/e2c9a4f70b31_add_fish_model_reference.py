"""add fishmodelreference table, seed it, and add the accuracy view

Revision ID: e2c9a4f70b31
Revises: d8b3f16c204e
Create Date: 2026-08-04 09:00:00.000000

Persists the known lengths of the physical fish models (the pipeline's
held-out validation set, never used for calibration) and exposes
`fish_model_measurement_accuracy` so measured-vs-known error is queryable
from Superset. Seed data is canonical in `views.KNOWN_FISH_MODELS`.

"""
# pylint: skip-file

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from fishsense_api.views import (
    DROP_FISH_MODEL_ACCURACY_VIEW_SQL,
    FISH_MODEL_ACCURACY_VIEW_SQL,
    KNOWN_FISH_MODELS,
)


# revision identifiers, used by Alembic.
revision: str = "e2c9a4f70b31"
down_revision: Union[str, Sequence[str], None] = "d8b3f16c204e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create + seed the reference table, then (re)create the accuracy view.

    Table creation is idempotent against `SQLModel.metadata.create_all` (the
    lifespan runs it first, and the model is in the registry). The seed is an
    upsert-by-absence so re-running never duplicates and never clobbers a
    length an operator has corrected by hand.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("fishmodelreference"):
        op.create_table(
            "fishmodelreference",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("known_length_m", sa.Float(), nullable=False),
            sa.Column("notes", sa.String(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("name", name="uq_fish_model_reference_name"),
        )
        op.create_index(
            "ix_fishmodelreference_name", "fishmodelreference", ["name"]
        )

    # Seed only the models that aren't present yet.
    existing = {
        row[0]
        for row in bind.execute(sa.text("SELECT name FROM fishmodelreference"))
    }
    to_insert = [m for m in KNOWN_FISH_MODELS if m["name"] not in existing]
    if to_insert:
        bind.execute(
            sa.text(
                "INSERT INTO fishmodelreference (name, known_length_m) "
                "VALUES (:name, :known_length_m)"
            ),
            to_insert,
        )

    op.execute(DROP_FISH_MODEL_ACCURACY_VIEW_SQL)
    op.execute(FISH_MODEL_ACCURACY_VIEW_SQL)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(DROP_FISH_MODEL_ACCURACY_VIEW_SQL)
    op.drop_index("ix_fishmodelreference_name", table_name="fishmodelreference")
    op.drop_table("fishmodelreference")
