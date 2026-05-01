# pylint: disable=unused-argument
"""Unit tests for select_next_high_priority_dive_for_headtail_preprocessing_activity."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.headtail_label import HeadTailLabel
from fishsense_api_sdk.models.species_label import SpeciesLabel
from fishsense_api_workflow_worker.activities import (
    select_next_high_priority_dive_for_headtail_preprocessing_activity as sut,
)


def _dive(dive_id: int, *, priority: str = "HIGH") -> Dive:
    return Dive(
        id=dive_id,
        name=f"dive-{dive_id}",
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority=priority,
        flip_dive_slate=False,
        camera_id=1,
        dive_slate_id=None,
    )


def _species(image_id: int, *, top_three: bool) -> SpeciesLabel:
    return SpeciesLabel(
        id=None,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=70,
        image_url=None,
        updated_at=None,
        completed=True,
        grouping=None,
        top_three_photos_of_group=top_three,
        slate_upside_down=None,
        laser_x=None,
        laser_y=None,
        laser_label=None,
        content_of_image=None,
        fish_measurable_category=None,
        fish_angle_category=None,
        fish_curved_category=None,
        label_studio_json={},
        image_id=image_id,
        user_id=None,
    )


def _headtail(image_id: int, *, completed: bool) -> HeadTailLabel:
    return HeadTailLabel(
        id=None,
        image_id=image_id,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=71,
        image_url=None,
        updated_at=None,
        completed=completed,
        head_x=None,
        head_y=None,
        tail_x=None,
        tail_y=None,
        label_studio_json={},
        user_id=None,
        superseded=False,
    )


def _make_fs(
    *,
    dives: List[Dive],
    species_by_dive: dict[int, List[SpeciesLabel]],
    headtail_by_dive: dict[int, List[HeadTailLabel]],
):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(return_value=dives)

    fs.labels = MagicMock()

    async def _get_species(dive_id):
        return species_by_dive.get(dive_id, [])

    async def _get_headtail(dive_id):
        return headtail_by_dive.get(dive_id, [])

    fs.labels.get_species_labels = AsyncMock(side_effect=_get_species)
    fs.labels.get_headtail_labels = AsyncMock(side_effect=_get_headtail)
    return fs


@pytest.mark.asyncio
async def test_picks_dive_with_top_three_species_and_incomplete_headtail(monkeypatch):
    fs = _make_fs(
        dives=[_dive(2), _dive(1), _dive(3, priority="LOW")],
        species_by_dive={
            1: [_species(101, top_three=True)],
            2: [_species(201, top_three=True)],
        },
        headtail_by_dive={
            1: [_headtail(101, completed=False)],
            2: [_headtail(201, completed=True)],
        },
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_headtail_preprocessing_activity
    )
    assert result == 1


@pytest.mark.asyncio
async def test_skips_dive_without_top_three_species(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1), _dive(2)],
        species_by_dive={
            1: [_species(101, top_three=False)],
            2: [_species(201, top_three=True)],
        },
        headtail_by_dive={2: []},
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_headtail_preprocessing_activity
    )
    assert result == 2


@pytest.mark.asyncio
async def test_skips_dive_when_all_top_three_have_completed_headtail(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1)],
        species_by_dive={1: [_species(101, top_three=True)]},
        headtail_by_dive={1: [_headtail(101, completed=True)]},
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_headtail_preprocessing_activity
    )
    assert result is None


@pytest.mark.asyncio
async def test_picks_dive_when_top_three_has_no_headtail_label(monkeypatch):
    # No headtail row at all -> incomplete.
    fs = _make_fs(
        dives=[_dive(1)],
        species_by_dive={1: [_species(101, top_three=True)]},
        headtail_by_dive={1: []},
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_headtail_preprocessing_activity
    )
    assert result == 1
