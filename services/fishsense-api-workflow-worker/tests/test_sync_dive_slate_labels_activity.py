# pylint: disable=unused-argument
"""Unit tests for sync_dive_slate_labels_for_label_studio_project_activity.

Three things this file pins down:
  1. `_parse_results` correctly pulls upside_down / reference_points /
     slate_rectangle / skipped_points out of a realistic LS payload.
  2. `compute_pdf_panel_aspect_ratio` returns w/h from page.rect, and
     `compute_pdf_panel_width_in_composite` applies it to scale.
  3. The activity bounds per-task work with `SYNC_CONCURRENCY` and
     fires one heartbeat per task — same regression guards the laser
     and headtail sync activity tests pin in their files.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pymupdf
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    sync_dive_slate_labels_for_label_studio_project_activity as sut,
    utils as sut_utils,
)


def _make_task(task_id: int, *, annotations: List[dict] | None = None) -> Any:
    return SimpleNamespace(
        id=task_id,
        annotators=[],
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
    fs.labels.get_dive_slate_label = AsyncMock(side_effect=_get)
    fs.labels.put_dive_slate_label = AsyncMock()
    fs.labels.get_sync_cursor = AsyncMock(side_effect=_get_cursor)
    fs.labels.put_sync_cursor = AsyncMock()
    return fs


def _make_ls_client(tasks: List[Any]):
    ls = MagicMock()
    ls.projects = MagicMock()
    ls.projects.get = MagicMock(return_value=SimpleNamespace(id=1))
    ls.tasks = MagicMock()
    ls.tasks.list = MagicMock(return_value=tasks)
    return ls


# ----------------------------- pure parser -----------------------------


def test_parse_results_extracts_all_fields():
    annotation = {
        "result": [
            {
                "from_name": "upside_down",
                "value": {"choices": ["Slate upside down"]},
            },
            {
                "from_name": "reference_points",
                "value": {"x": 50.0, "y": 25.0, "keypointlabels": ["Reference Point"]},
                "original_width": 1000,
                "original_height": 500,
            },
            {
                "from_name": "reference_points",
                "value": {"x": 75.0, "y": 30.0, "keypointlabels": ["Reference Point"]},
                "original_width": 1000,
                "original_height": 500,
            },
            {
                "from_name": "slate",
                "value": {
                    "x": 60.0,
                    "y": 10.0,
                    "width": 20.0,
                    "height": 30.0,
                    "rectanglelabels": ["Slate"],
                },
                "original_width": 1000,
                "original_height": 500,
            },
            {
                "from_name": "skipped_points",
                "value": {"text": ["1", "3", "5"]},
            },
        ]
    }

    parsed = sut._parse_results(annotation)
    assert parsed["upside_down"] is True
    assert parsed["reference_points"] == [(500.0, 125.0), (750.0, 150.0)]
    assert parsed["slate_rectangle"] == [(600.0, 50.0), (800.0, 200.0)]
    assert parsed["skipped_points"] == [0, 2, 4]
    assert parsed["original_height"] == 500.0


def test_parse_results_handles_minimal_annotation():
    parsed = sut._parse_results({"result": []})
    assert parsed["upside_down"] is None
    assert parsed["reference_points"] == []
    assert parsed["slate_rectangle"] is None
    assert parsed["skipped_points"] is None
    assert parsed["original_height"] is None


def test_parse_results_upside_down_false_when_choice_is_other():
    annotation = {
        "result": [
            {"from_name": "upside_down", "value": {"choices": ["Slate right side up"]}}
        ]
    }
    parsed = sut._parse_results(annotation)
    assert parsed["upside_down"] is False


# --------------------------- pdf aspect ratio --------------------------


def _make_synthetic_pdf(width_pts: float, height_pts: float) -> bytes:
    doc = pymupdf.open()
    doc.new_page(width=width_pts, height=height_pts)
    out = doc.tobytes()
    doc.close()
    return out


def test_compute_pdf_panel_aspect_ratio_matches_page_rect():
    pdf_bytes = _make_synthetic_pdf(216.0, 108.0)  # 2:1
    aspect = sut.compute_pdf_panel_aspect_ratio(pdf_bytes)
    assert aspect == pytest.approx(2.0, rel=1e-6)


def test_compute_pdf_panel_width_in_composite_scales_by_original_height():
    # 2:1 aspect, composite height 500 → panel width 1000.
    pw = sut.compute_pdf_panel_width_in_composite(2.0, 500.0)
    assert pw == pytest.approx(1000.0, rel=1e-6)


# ----------------------- activity-shape guards ------------------------


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

    fs.labels.get_dive_slate_label = AsyncMock(side_effect=_slow_get)

    monkeypatch.setattr(sut_utils, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut_utils, "get_ls_client", lambda: ls)

    await ActivityEnvironment().run(
        sut.sync_dive_slate_labels_for_label_studio_project_activity, 1
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
        sut.sync_dive_slate_labels_for_label_studio_project_activity, 1
    )

    assert len(heartbeats) == n_tasks
