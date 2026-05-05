"""add label studio sync cursor

Revision ID: 299428e39cb9
Revises: e1fc97743091
Create Date: 2026-05-01 00:00:00.000000

"""
# pylint: skip-file

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '299428e39cb9'
down_revision: Union[str, Sequence[str], None] = 'e1fc97743091'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Idempotent against `SQLModel.metadata.create_all`: the FastAPI
    lifespan runs `create_all` before alembic upgrade, so on a
    fresh-bootstrap deploy this table already exists by the time the
    migration runs. Skip the DDL when the table is present.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table('labelstudiosynccursor'):
        return
    op.create_table(
        'labelstudiosynccursor',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('kind', sa.String(), nullable=False),
        sa.Column('label_studio_project_id', sa.Integer(), nullable=False),
        sa.Column('last_synced_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'kind',
            'label_studio_project_id',
            name='uq_labelstudiosynccursor_kind_project',
        ),
    )
    op.create_index(
        op.f('ix_labelstudiosynccursor_kind'),
        'labelstudiosynccursor',
        ['kind'],
        unique=False,
    )
    op.create_index(
        op.f('ix_labelstudiosynccursor_label_studio_project_id'),
        'labelstudiosynccursor',
        ['label_studio_project_id'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f('ix_labelstudiosynccursor_label_studio_project_id'),
        table_name='labelstudiosynccursor',
    )
    op.drop_index(
        op.f('ix_labelstudiosynccursor_kind'),
        table_name='labelstudiosynccursor',
    )
    op.drop_table('labelstudiosynccursor')
