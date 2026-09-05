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
from botocore.exceptions import ClientError
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
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

    parsed = sut._parse_results(annotation)  # pylint: disable=protected-access
    assert "upside_down" not in parsed  # removed 2026-07-31
    assert parsed["reference_points"] == [(500.0, 125.0), (750.0, 150.0)]
    assert parsed["slate_rectangle"] == [(600.0, 50.0), (800.0, 200.0)]
    assert parsed["skipped_points"] == [0, 2, 4]
    assert parsed["original_height"] == 500.0
    assert parsed["original_width"] == 1000.0


def test_parse_results_handles_minimal_annotation():
    parsed = sut._parse_results({"result": []})  # pylint: disable=protected-access
    assert not parsed["reference_points"]
    assert parsed["slate_rectangle"] is None
    assert parsed["skipped_points"] is None
    assert parsed["original_height"] is None
    assert parsed["original_width"] is None


def test_parse_results_ignores_removed_upside_down_control():
    # A stale project may still emit an `upside_down` result; it must be a no-op.
    annotation = {
        "result": [
            {"from_name": "upside_down", "value": {"choices": ["Slate upside down"]}}
        ]
    }
    parsed = sut._parse_results(annotation)  # pylint: disable=protected-access
    assert "upside_down" not in parsed
    assert not parsed["reference_points"]


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


# ------------------- panel-offset fail-hard + bounds (Bug 1) -------------------


def _full_slate_label(image_id: int = 10, ls_id: int = 1) -> DiveSlateLabel:
    return DiveSlateLabel(
        id=ls_id,
        label_studio_task_id=ls_id,
        label_studio_project_id=66,
        image_url=None,
        upside_down=None,
        reference_points=None,
        slate_rectangle=None,
        skipped_points=None,
        updated_at=None,
        completed=False,
        superseded=False,
        label_studio_json=None,
        image_id=image_id,
        user_id=None,
    )


def _geometry_annotation(composite_x: float, y: float, ow: float, oh: float) -> dict:
    """One reference point at composite pixel (composite_x, y) on an ow×oh canvas."""
    return {
        "result": [
            {
                "from_name": "reference_points",
                "value": {
                    "x": composite_x / ow * 100.0,
                    "y": y / oh * 100.0,
                    "keypointlabels": ["Reference Point"],
                },
                "original_width": ow,
                "original_height": oh,
            }
        ]
    }


def _make_update_fs(slate_label: DiveSlateLabel, *, slate_id: int | None):
    fs = MagicMock()

    async def _get_label(label_studio_id):
        return slate_label

    fs.labels = MagicMock()
    fs.labels.get_dive_slate_label = AsyncMock(side_effect=_get_label)
    fs.labels.put_dive_slate_label = AsyncMock()

    async def _get_image(image_id):
        return SimpleNamespace(dive_id=5)

    async def _get_dive(dive_id):
        return SimpleNamespace(dive_slate_id=slate_id)

    fs.images = MagicMock()
    fs.images.get = AsyncMock(side_effect=_get_image)
    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(side_effect=_get_dive)
    return fs


def _make_exchange(*, pdf_bytes: bytes | None):
    ex = MagicMock()
    if pdf_bytes is None:
        ex.download_slate_pdf = AsyncMock(
            side_effect=ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )
        )
    else:
        ex.download_slate_pdf = AsyncMock(return_value=pdf_bytes)
    return ex


def test_assert_in_frame_rejects_out_of_bounds():
    sut._assert_in_frame([(50.0, 1.0)], 100.0, task_id=1, slate_id=2)  # in frame, no raise
    sut._assert_in_frame([(100.0, 1.0)], 100.0, task_id=1, slate_id=2)  # right edge OK
    with pytest.raises(ValueError):
        sut._assert_in_frame([(150.0, 1.0)], 100.0, task_id=1, slate_id=2)
    with pytest.raises(ValueError):
        sut._assert_in_frame([(-5.0, 1.0)], 100.0, task_id=1, slate_id=2)


@pytest.mark.asyncio
async def test_update_slate_label_subtracts_panel_offset_and_persists():
    # 2:1 PDF, composite height 100 -> panel width 200; canvas 300 wide ->
    # photo width 100. A point at composite x=250 lands at photo x=50.
    pdf = _make_synthetic_pdf(216.0, 108.0)  # aspect 2.0
    label = _full_slate_label()
    fs = _make_update_fs(label, slate_id=7)
    exchange = _make_exchange(pdf_bytes=pdf)
    task = _make_task(1, annotations=[_geometry_annotation(250.0, 50.0, 300.0, 100.0)])

    await sut._update_slate_label(
        fs, task, exchange=exchange, aspect_cache={}, image_to_slate={}
    )

    fs.labels.put_dive_slate_label.assert_awaited_once()
    (persisted_image_id, persisted), _ = fs.labels.put_dive_slate_label.call_args
    assert persisted_image_id == 10
    assert len(persisted.reference_points) == 1
    px, py = persisted.reference_points[0]
    assert px == pytest.approx(50.0)
    assert py == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_update_slate_label_raises_and_skips_persist_when_pdf_missing():
    # The old fallback silently persisted composite-space coords with a 0px
    # shift. It must now fail the label instead — no put, exception propagates.
    label = _full_slate_label()
    fs = _make_update_fs(label, slate_id=7)
    exchange = _make_exchange(pdf_bytes=None)  # NoSuchKey
    task = _make_task(1, annotations=[_geometry_annotation(250.0, 50.0, 300.0, 100.0)])

    with pytest.raises(ValueError, match="object store"):
        await sut._update_slate_label(
            fs, task, exchange=exchange, aspect_cache={}, image_to_slate={}
        )

    fs.labels.put_dive_slate_label.assert_not_called()


@pytest.mark.asyncio
async def test_update_slate_label_raises_when_slate_unresolvable():
    label = _full_slate_label()
    fs = _make_update_fs(label, slate_id=None)  # dive has no dive_slate_id
    exchange = _make_exchange(pdf_bytes=_make_synthetic_pdf(216.0, 108.0))
    task = _make_task(1, annotations=[_geometry_annotation(250.0, 50.0, 300.0, 100.0)])

    with pytest.raises(ValueError, match="dive_slate_id"):
        await sut._update_slate_label(
            fs, task, exchange=exchange, aspect_cache={}, image_to_slate={}
        )

    fs.labels.put_dive_slate_label.assert_not_called()


@pytest.mark.asyncio
async def test_update_slate_label_raises_when_shift_lands_out_of_frame():
    # Wrong-template panel width: composite x=250 needs a 200px shift, but a
    # too-large panel (aspect 3.0 -> 300px) overshoots to -50, out of frame.
    pdf = _make_synthetic_pdf(324.0, 108.0)  # aspect 3.0 -> panel 300
    label = _full_slate_label()
    fs = _make_update_fs(label, slate_id=7)
    exchange = _make_exchange(pdf_bytes=pdf)
    task = _make_task(1, annotations=[_geometry_annotation(250.0, 50.0, 300.0, 100.0)])

    with pytest.raises(ValueError, match="outside"):
        await sut._update_slate_label(
            fs, task, exchange=exchange, aspect_cache={}, image_to_slate={}
        )

    fs.labels.put_dive_slate_label.assert_not_called()


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
