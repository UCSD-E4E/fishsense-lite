"""Sync cursor model for incremental Label Studio label syncs."""

from __future__ import annotations

from datetime import datetime

from fishsense_api_sdk.models.model_base import ModelBase


class LabelStudioSyncCursor(ModelBase):
    # pylint: disable=R0801
    """Per-(kind, project) high-water mark for Label Studio sync."""

    id: int | None = None
    kind: str
    label_studio_project_id: int
    last_synced_at: datetime | None = None
