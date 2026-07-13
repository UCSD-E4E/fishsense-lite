# pylint: disable=unused-argument
"""Unit tests for sync_species_labels_for_label_studio_project_activity.

Three things this file pins down:
  1. `_parse_results` correctly extracts every species annotation
     field still present in the post-2026-05-05 XML (grouping,
     exclude→top_three, content/measurable/angle/curve taxonomies).
     The laser keypoint and "Slate upside down" controls were dropped
     from the XML so this activity no longer reads them.
  2. `_apply_parsed` only writes fields the annotation actually
     specified — a re-sync that drops a section doesn't clobber the
     stored value.
  3. The activity bounds per-task work with `SYNC_CONCURRENCY` and
     fires one heartbeat per task (same regression guards as the
     laser/headtail/dive-slate sync activity tests).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.species_label import SpeciesLabel
from fishsense_api_workflow_worker.activities import (
    sync_species_labels_for_label_studio_project_activity as sut,
    utils as sut_utils,
)


def _make_task(
    task_id: int,
    *,
    annotations: List[dict] | None = None,
    annotators: List[int] | None = None,
) -> Any:
    return SimpleNamespace(
        id=task_id,
        annotators=annotators or [],
        annotations=annotations or [],
        is_labeled=False,
        updated_at="2026-05-01T00:00:00Z",
        json=lambda: "{}",
    )


def _make_fs_client(label_lookup, *, cursor=None):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    async def _get(label_studio_id):
        return label_lookup.get(label_studio_id)

    async def _get_cursor(kind, project_id):
        return cursor

    fs.labels = MagicMock()
    fs.labels.get_species_label = AsyncMock(side_effect=_get)
    fs.labels.put_species_label = AsyncMock()
    fs.labels.get_sync_cursor = AsyncMock(side_effect=_get_cursor)
    fs.labels.put_sync_cursor = AsyncMock()
    # Default: the annotator isn't user-synced yet -> None.
    fs.users = MagicMock()
    fs.users.get_by_label_studio_id = AsyncMock(return_value=None)
    return fs


def _make_ls_client(tasks: List[Any]):
    ls = MagicMock()
    ls.projects = MagicMock()
    ls.projects.get = MagicMock(return_value=SimpleNamespace(id=1))
    ls.tasks = MagicMock()
    ls.tasks.list = MagicMock(return_value=tasks)
    return ls


def _empty_species_label(image_id: int) -> SpeciesLabel:
    return SpeciesLabel(
        id=None,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=70,
        image_url=None,
        updated_at=None,
        completed=False,
        grouping=None,
        top_three_photos_of_group=None,
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


# ----------------------------- pure parser -----------------------------


def test_parse_results_extracts_all_fields():
    annotation = {
        "result": [
            {"from_name": "grouping", "value": {"choices": ["Part of previous group"]}},
            {"from_name": "exclude", "value": {"choices": ["Top 3 photos of group"]}},
            {
                "from_name": "species",
                "value": {"taxonomy": [["Slate", "Laser on slate"]]},
            },
            {
                "from_name": "measurable",
                "value": {"taxonomy": [["Measurable"]]},
            },
            {"from_name": "fishAngles", "value": {"taxonomy": [["Side"]]}},
            {"from_name": "fishCurve", "value": {"taxonomy": [["No Curve"]]}},
        ]
    }

    parsed = sut._parse_results(annotation)  # pylint: disable=protected-access

    assert parsed["grouping"] == "Part of previous group"
    assert parsed["top_three_photos_of_group"] is True
    assert parsed["content_of_image"] == "Slate, Laser on slate"
    assert parsed["fish_measurable_category"] == "Measurable"
    assert parsed["fish_angle_category"] == "Side"
    assert parsed["fish_curved_category"] == "No Curve"


def test_parse_results_handles_minimal_annotation():
    parsed = sut._parse_results({"result": []})  # pylint: disable=protected-access

    assert parsed["grouping"] is None
    assert parsed["top_three_photos_of_group"] is None
    assert parsed["content_of_image"] is None
    assert parsed["fish_measurable_category"] is None
    assert parsed["fish_angle_category"] is None
    assert parsed["fish_curved_category"] is None


def test_parse_results_recognizes_negative_top_three_choice():
    annotation = {
        "result": [
            {"from_name": "exclude", "value": {"choices": ["Skip this group"]}},
        ]
    }
    parsed = sut._parse_results(annotation)  # pylint: disable=protected-access
    # exclude is True only when literally "Top 3 photos of group" —
    # anything else is False (we still got a choice, just not the
    # affirmative one).
    assert parsed["top_three_photos_of_group"] is False


# ----------------------------- _apply_parsed ----------------------------


def test_apply_parsed_only_overwrites_specified_fields():
    label = _empty_species_label(image_id=1)
    label.grouping = "Part of previous group"  # pre-existing
    label.content_of_image = "Fish, Hogfish"  # pre-existing

    parsed = sut._parse_results(  # pylint: disable=protected-access
        {"result": [{"from_name": "fishAngles", "value": {"taxonomy": [["Top"]]}}]}
    )

    sut._apply_parsed(label, parsed)  # pylint: disable=protected-access

    # Only fishAngles was specified; grouping and content must survive.
    assert label.fish_angle_category == "Top"
    assert label.grouping == "Part of previous group"
    assert label.content_of_image == "Fish, Hogfish"


def test_apply_parsed_does_not_clobber_dropped_fields_on_historical_rows():
    """The 2026-05-05 species XML refresh dropped the laser keypoint
    and "Slate upside down" controls. `_apply_parsed` and
    `_parse_results` were stripped of those extraction paths
    accordingly. Pinning the regression: a historical SpeciesLabel
    row that has `laser_x` / `laser_y` / `laser_label` /
    `slate_upside_down` populated must survive a re-sync against the
    new XML *unchanged*. A future "helpful" zeroing of those fields
    would silently invalidate every measurement that consumed them
    before the refresh.
    """
    label = _empty_species_label(image_id=1)
    # Pre-existing values from a row populated under the old XML.
    label.laser_x = 1234.5
    label.laser_y = 678.9
    label.laser_label = "Red Laser"
    label.slate_upside_down = True

    # New-XML annotation: only the still-supported fields.
    parsed = sut._parse_results(  # pylint: disable=protected-access
        {
            "result": [
                {
                    "from_name": "grouping",
                    "value": {"choices": ["Part of previous group"]},
                },
                {
                    "from_name": "species",
                    "value": {"taxonomy": [["Fish", "Hogfish"]]},
                },
            ]
        }
    )

    sut._apply_parsed(label, parsed)  # pylint: disable=protected-access

    # The dropped-XML fields must still hold their historical values.
    assert label.laser_x == 1234.5
    assert label.laser_y == 678.9
    assert label.laser_label == "Red Laser"
    assert label.slate_upside_down is True
    # And the new-XML fields landed.
    assert label.grouping == "Part of previous group"
    assert label.content_of_image == "Fish, Hogfish"


# ----------------------- activity-shape guards ------------------------


@pytest.mark.asyncio
async def test_skips_tasks_with_no_existing_label(monkeypatch):
    tasks = [_make_task(i) for i in range(3)]
    fs = _make_fs_client(label_lookup={})  # every get returns None
    ls = _make_ls_client(tasks)

    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.sync_species_labels_for_label_studio_project_activity, 1
    )

    assert fs.labels.get_species_label.await_count == 3
    fs.labels.put_species_label.assert_not_called()


@pytest.mark.asyncio
async def test_unmapped_annotator_does_not_crash_sync(monkeypatch):
    """A task annotated by someone not yet user-synced (get_by_label_studio_id
    -> None) must sync without crashing; user_id is simply left unset. Happens
    during a Label-Studio-instance transition."""
    task = _make_task(1, annotators=[999])
    label = _empty_species_label(11)
    fs = _make_fs_client(label_lookup={1: label})  # annotator maps to None
    ls = _make_ls_client([task])

    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.sync_species_labels_for_label_studio_project_activity, 1
    )

    fs.users.get_by_label_studio_id.assert_awaited_once_with(999)
    written = fs.labels.put_species_label.await_args_list
    assert len(written) == 1
    assert written[0].args[1].user_id is None


@pytest.mark.asyncio
async def test_per_task_concurrency_is_bounded_by_semaphore(monkeypatch):
    n_tasks = 50
    tasks = [_make_task(i) for i in range(n_tasks)]
    fs = _make_fs_client(label_lookup={})
    ls = _make_ls_client(tasks)

    in_flight = 0
    peak_in_flight = 0

    async def _slow_get(label_studio_id):
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        in_flight -= 1
        return None

    fs.labels.get_species_label = AsyncMock(side_effect=_slow_get)

    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.sync_species_labels_for_label_studio_project_activity, 1
    )

    assert peak_in_flight <= sut.SYNC_CONCURRENCY, (
        f"peak concurrency was {peak_in_flight}, "
        f"expected <= {sut.SYNC_CONCURRENCY}"
    )


@pytest.mark.asyncio
async def test_heartbeat_fires_per_completed_task(monkeypatch):
    n_tasks = 5
    tasks = [_make_task(i) for i in range(n_tasks)]

    fs = _make_fs_client(label_lookup={})
    ls = _make_ls_client(tasks)

    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)

    heartbeats: List[tuple] = []

    env = ActivityEnvironment()
    env.on_heartbeat = lambda *args: heartbeats.append(args)

    await env.run(
        sut.sync_species_labels_for_label_studio_project_activity, 1
    )

    assert len(heartbeats) == n_tasks


@pytest.mark.asyncio
async def test_writes_parsed_fields_when_annotation_present(monkeypatch):
    annotation = {
        "result": [
            {"from_name": "grouping", "value": {"choices": ["Part of previous group"]}},
            {
                "from_name": "species",
                "value": {"taxonomy": [["Reef fish", "Yellowtail Snapper"]]},
            },
        ]
    }
    tasks = [_make_task(101, annotations=[annotation])]
    label = _empty_species_label(image_id=42)
    fs = _make_fs_client(label_lookup={101: label})
    ls = _make_ls_client(tasks)

    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.sync_species_labels_for_label_studio_project_activity, 1
    )

    fs.labels.put_species_label.assert_awaited_once()
    written = fs.labels.put_species_label.await_args.args[1]
    assert written.grouping == "Part of previous group"
    assert written.content_of_image == "Reef fish, Yellowtail Snapper"
