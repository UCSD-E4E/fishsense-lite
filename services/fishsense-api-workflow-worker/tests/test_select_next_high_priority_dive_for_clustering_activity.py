# pylint: disable=unused-argument
"""Unit tests for select_next_high_priority_dive_for_clustering_activity.

Cohort predicate moved server-side; see
`services/fishsense-api/tests/test_select_next_dive_endpoints.py`. The
activity is now a passthrough to the SDK.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    select_next_high_priority_dive_for_clustering_activity as sut,
)


def _make_fs(*, dive_id: int | None):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.dives = MagicMock()
    fs.dives.select_next_for_dive_frame_clustering = AsyncMock(return_value=dive_id)
    return fs


@pytest.mark.asyncio
async def test_passes_through_dive_id_from_sdk(monkeypatch):
    fs = _make_fs(dive_id=42)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_clustering_activity
    )

    assert result == 42
    fs.dives.select_next_for_dive_frame_clustering.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_returns_none_when_sdk_returns_none(monkeypatch):
    fs = _make_fs(dive_id=None)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_clustering_activity
    )

    assert result is None
