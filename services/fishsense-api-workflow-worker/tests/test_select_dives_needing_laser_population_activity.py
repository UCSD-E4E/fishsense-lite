# pylint: disable=protected-access
"""Unit test for select_dives_needing_laser_population_activity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    select_dives_needing_laser_population_activity as sut,
)


@pytest.mark.asyncio
async def test_returns_cohort_from_sdk(monkeypatch):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.dives = MagicMock()
    fs.dives.get_dives_needing_laser_population = AsyncMock(return_value=[5, 9])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_dives_needing_laser_population_activity
    )
    assert result == [5, 9]
