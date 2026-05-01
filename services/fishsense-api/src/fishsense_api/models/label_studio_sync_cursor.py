"""Sync cursor for incremental Label Studio label syncs.

A row records the highest `task.updated_at` we've successfully synced for
a given (kind, label_studio_project_id) pair. The api-workflow-worker
reads this on each scheduled run and skips Label Studio tasks whose
`updated_at` is at or below the cursor.

The cursor is only advanced after a project's full sync completes
without exception (TaskGroup all-or-nothing semantics) so a partial
failure replays the same range on the next run; per-task PUTs are
upserts, so replay is safe.
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import DateTime, Field, UniqueConstraint

from fishsense_api.models.model_base import ModelBase


class LabelStudioSyncCursor(ModelBase, table=True):
    # pylint: disable=R0801
    """Per-(kind, project) high-water mark for Label Studio sync."""

    __table_args__ = (
        UniqueConstraint(
            "kind",
            "label_studio_project_id",
            name="uq_labelstudiosynccursor_kind_project",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    kind: str = Field(index=True)
    label_studio_project_id: int = Field(index=True)
    last_synced_at: datetime | None = Field(
        sa_type=DateTime(timezone=True), default=None
    )
