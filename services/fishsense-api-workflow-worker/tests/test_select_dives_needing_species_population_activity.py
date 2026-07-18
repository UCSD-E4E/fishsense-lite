# pylint: disable=unused-argument
"""Unit tests for select_dives_needing_species_population_activity.

Superseded-aware cohort predicate lives server-side; see
`services/fishsense-api/tests/test_select_next_dive_endpoints.py`. The
activity is a passthrough to the SDK's list-returning method.
"""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    select_dives_needing_species_population_activity as sut,
)


def _make_fs(*, dive_ids: List[int]):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.dives = MagicMock()
    fs.dives.get_dives_needing_species_population = AsyncMock(return_value=dive_ids)
    return fs


@pytest.mark.asyncio
async def test_passes_through_dive_ids_from_sdk(monkeypatch):
    fs = _make_fs(dive_ids=[3, 7, 11])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_dives_needing_species_population_activity
    )

    assert result == [3, 7, 11]
    fs.dives.get_dives_needing_species_population.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_returns_empty_list_when_cohort_empty(monkeypatch):
    fs = _make_fs(dive_ids=[])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_dives_needing_species_population_activity
    )

    assert result == []
