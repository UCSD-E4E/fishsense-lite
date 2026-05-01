# pylint: disable=unused-argument
"""Unit tests for select_next_high_priority_dive_for_slate_preprocessing_activity."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.dive import Dive
from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
from fishsense_api_sdk.models.species_label import SpeciesLabel
from fishsense_api_workflow_worker.activities import (
    select_next_high_priority_dive_for_slate_preprocessing_activity as sut,
)


def _dive(dive_id: int, *, priority: str = "HIGH", dive_slate_id: int | None = 7) -> Dive:
    return Dive(
        id=dive_id,
        name=f"dive-{dive_id}",
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority=priority,
        flip_dive_slate=False,
        camera_id=1,
        dive_slate_id=dive_slate_id,
    )


def _species(image_id: int, *, content: str | None) -> SpeciesLabel:
    return SpeciesLabel(
        id=None,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=70,
        image_url=None,
        updated_at=None,
        completed=True,
        grouping=None,
        top_three_photos_of_group=None,
        slate_upside_down=None,
        laser_x=None,
        laser_y=None,
        laser_label=None,
        content_of_image=content,
        fish_measurable_category=None,
        fish_angle_category=None,
        fish_curved_category=None,
        label_studio_json={},
        image_id=image_id,
        user_id=None,
    )


def _slate_label(image_id: int, *, completed: bool) -> DiveSlateLabel:
    return DiveSlateLabel(
        id=None,
        image_id=image_id,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=66,
        image_url=None,
        updated_at=None,
        completed=completed,
        upside_down=None,
        reference_points=None,
        slate_rectangle=None,
        skipped_points=None,
        label_studio_json={},
        user_id=None,
    )


def _make_fs(
    *,
    dives: List[Dive],
    species_by_dive: dict[int, List[SpeciesLabel]],
    slate_by_dive: dict[int, List[DiveSlateLabel]],
):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(return_value=dives)

    fs.labels = MagicMock()

    async def _get_species(dive_id):
        return species_by_dive.get(dive_id, [])

    async def _get_slate(dive_id):
        return slate_by_dive.get(dive_id, [])

    fs.labels.get_species_labels = AsyncMock(side_effect=_get_species)
    fs.labels.get_dive_slate_labels = AsyncMock(side_effect=_get_slate)
    return fs


SLATE = "Slate, Laser on slate"


@pytest.mark.asyncio
async def test_picks_dive_with_slate_marked_species_and_incomplete_slate(monkeypatch):
    fs = _make_fs(
        dives=[_dive(2), _dive(1), _dive(3, dive_slate_id=None)],
        species_by_dive={
            1: [_species(101, content=SLATE)],
            2: [_species(201, content=SLATE)],
        },
        slate_by_dive={
            1: [_slate_label(101, completed=False)],
            2: [_slate_label(201, completed=True)],
        },
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_slate_preprocessing_activity
    )
    assert result == 1


@pytest.mark.asyncio
async def test_skips_dive_without_dive_slate_id(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1, dive_slate_id=None), _dive(2)],
        species_by_dive={2: [_species(201, content=SLATE)]},
        slate_by_dive={2: []},
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_slate_preprocessing_activity
    )
    assert result == 2


@pytest.mark.asyncio
async def test_skips_dive_without_slate_marked_species(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1)],
        species_by_dive={1: [_species(101, content="Fish")]},
        slate_by_dive={},
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_slate_preprocessing_activity
    )
    assert result is None


@pytest.mark.asyncio
async def test_skips_dive_when_all_slate_labels_completed(monkeypatch):
    fs = _make_fs(
        dives=[_dive(1)],
        species_by_dive={1: [_species(101, content=SLATE)]},
        slate_by_dive={1: [_slate_label(101, completed=True)]},
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    result = await ActivityEnvironment().run(
        sut.select_next_high_priority_dive_for_slate_preprocessing_activity
    )
    assert result is None
