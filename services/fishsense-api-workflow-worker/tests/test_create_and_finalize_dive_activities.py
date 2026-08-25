"""`create_dive_activity` and `finalize_dive_activity` — the commit protocol.

Ingest writes a dive in two posts, and the pair is a two-phase commit against a
table that has no transactions across activities:

  * **create** writes the dive at `priority=LOW`, *whatever the request asked
    for*. LOW keeps it out of every hourly cohort, so a half-ingested dive
    cannot be picked up and processed.
  * **finalize** flips it to the requested priority — but only if every listed
    frame landed. **Priority is the commit flag.**

That is what makes a crashed or retried ingest safe: the dive exists, its images
exist, and the pipeline ignores all of it until someone can say the set is
complete.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

FOLDER = "2024.06.20.REEF/082929_FishModels_FSL07"
T0 = datetime(2024, 8, 21, 8, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2024, 8, 21, 9, 30, 0, tzinfo=timezone.utc)


def _request(**kwargs):
    from fishsense_shared.ingest_contracts import IngestDiveRequest

    kwargs.setdefault("dive_path", FOLDER)
    kwargs.setdefault("self_calibrates", True)
    return IngestDiveRequest(**kwargs)


def _preflight(**kwargs):
    from fishsense_shared.ingest_contracts import IngestPreflight, PreflightImage

    kwargs.setdefault("dive_path", f"/fishsense_data/REEF/data/{FOLDER}")
    kwargs.setdefault("resolved_camera_id", 7)
    kwargs.setdefault(
        "images",
        [
            PreflightImage(path=f"{FOLDER}/A.ORF", size=1, taken_datetime=T0),
            PreflightImage(path=f"{FOLDER}/B.ORF", size=1, taken_datetime=T1),
        ],
    )
    return IngestPreflight(**kwargs)


def _fs(dive_id=412, images=None, lookup=None):
    fs = MagicMock()
    fs.dives.post = AsyncMock(return_value=dive_id)
    fs.images.get = AsyncMock(return_value=images or [])
    fs.images.lookup_checksums = AsyncMock(return_value=lookup or {})
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=False)
    return fs


def _image(path, checksum):
    img = MagicMock()
    img.path = path
    img.checksum = checksum
    return img


async def _create(request, preflight, fs, monkeypatch):
    from fishsense_api_workflow_worker.activities import create_dive_activity as sut

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    return await ActivityEnvironment().run(
        sut.create_dive_activity, request, preflight
    )


async def _finalize(dive_id, request, totals, fs, monkeypatch):
    from fishsense_api_workflow_worker.activities import finalize_dive_activity as sut

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    return await ActivityEnvironment().run(
        sut.finalize_dive_activity, dive_id, request, totals
    )


def _totals(**kwargs):
    from fishsense_api_workflow_worker.activities.finalize_dive_activity import (
        IngestTotals,
    )

    kwargs.setdefault("total", 2)
    kwargs.setdefault("registered", 2)
    kwargs.setdefault("skipped_existing", 0)
    kwargs.setdefault("rejected", [])
    kwargs.setdefault("max_taken_datetime", T1)
    return IngestTotals(**kwargs)


# ── create: LOW is not negotiable ─────────────────────────────────────


async def test_creates_the_dive_at_low_even_when_high_was_requested(monkeypatch):
    """The whole safety property. HIGH is what the hourly cohorts select on, so
    a dive created HIGH before its images land would be picked up mid-ingest and
    processed against a partial set."""
    fs = _fs()

    dive_id = await _create(_request(priority="HIGH"), _preflight(), fs, monkeypatch)

    assert dive_id == 412
    (dive,), _ = fs.dives.post.call_args
    assert dive.priority.value == "LOW"


async def test_seeds_a_provisional_dive_datetime_from_preflight(monkeypatch):
    """`dive_datetime` is NOT NULL, so create needs a value before any frame has
    been hashed. Preflight already read every header, so use its max — finalize
    replaces it with the scan's."""
    fs = _fs()

    await _create(_request(), _preflight(), fs, monkeypatch)

    (dive,), _ = fs.dives.post.call_args
    assert dive.dive_datetime == T1


async def test_defaults_the_name_to_the_leaf_directory(monkeypatch):
    """`Dive.name` feeds the per-dive Label Studio project title, so an unnamed
    dive gets a title of just its id."""
    fs = _fs()

    await _create(_request(), _preflight(), fs, monkeypatch)

    (dive,), _ = fs.dives.post.call_args
    assert dive.name == "082929_FishModels_FSL07"


async def test_an_explicit_name_wins(monkeypatch):
    fs = _fs()

    await _create(_request(dive_name="Reef dive 3"), _preflight(), fs, monkeypatch)

    (dive,), _ = fs.dives.post.call_args
    assert dive.name == "Reef dive 3"


async def test_carries_the_camera_and_calibration_intent(monkeypatch):
    fs = _fs()

    await _create(
        _request(self_calibrates=False, calibration_dive_id=61, dive_slate_id=3),
        _preflight(),
        fs,
        monkeypatch,
    )

    (dive,), _ = fs.dives.post.call_args
    assert dive.camera_id == 7
    assert dive.calibration_dive_id == 61
    assert dive.dive_slate_id == 3


async def test_a_self_calibrating_dive_has_no_calibration_link(monkeypatch):
    """NULL means "self-calibrate". Writing a link here would make the dive
    borrow a calibration it does not need — own-wins resolution would ignore it,
    but the row would be lying."""
    fs = _fs()

    await _create(_request(self_calibrates=True), _preflight(), fs, monkeypatch)

    (dive,), _ = fs.dives.post.call_args
    assert dive.calibration_dive_id is None


# ── finalize: priority is the commit flag ─────────────────────────────


async def test_promotes_to_the_requested_priority_when_everything_landed(
    monkeypatch,
):
    fs = _fs()

    report = await _finalize(412, _request(priority="HIGH"), _totals(), fs, monkeypatch)

    (dive,), _ = fs.dives.post.call_args
    assert dive.priority.value == "HIGH"
    assert dive.dive_datetime == T1
    assert report.committed is True
    assert report.dive_id == 412


async def test_refuses_when_a_frame_was_rejected(monkeypatch):
    """A partially-ingested dive must never enter the pipeline. Non-retryable:
    a rejected frame is a data problem, and retrying re-reads the same bytes to
    the same conclusion."""
    from fishsense_shared.ingest_contracts import RejectedImage

    fs = _fs()
    totals = _totals(
        registered=1, rejected=[RejectedImage(path="x", reason="no timestamp")]
    )

    with pytest.raises(ApplicationError) as excinfo:
        await _finalize(412, _request(), totals, fs, monkeypatch)

    assert excinfo.value.non_retryable
    assert fs.dives.post.await_count == 0


async def test_refuses_when_the_counts_do_not_add_up(monkeypatch):
    """registered + skipped must equal total. A gap means a frame was neither
    written nor recognised as already present — silence rather than a rejection,
    which is worse."""
    fs = _fs()

    with pytest.raises(ApplicationError) as excinfo:
        await _finalize(412, _request(), _totals(registered=1), fs, monkeypatch)

    assert excinfo.value.non_retryable
    assert fs.dives.post.await_count == 0


async def test_a_dive_that_was_entirely_skipped_still_commits(monkeypatch):
    """Re-running a completed ingest is a no-op that must still succeed —
    otherwise the only way to re-verify a dive is to make it fail."""
    fs = _fs()

    report = await _finalize(
        412, _request(), _totals(registered=0, skipped_existing=2), fs, monkeypatch
    )

    assert report.committed is True
    assert report.skipped_existing == 2


# ── finalize: content overlap ─────────────────────────────────────────


async def test_reports_content_overlap_with_an_existing_dive(monkeypatch):
    """Layer 2 of duplicate detection, and the reason it runs HERE: it needs
    every frame's checksum, which only exists once the scan has written the
    rows. Containment is |new ∩ existing| / |new| over content hashes, so it is
    immune to filenames and ordering — the property the legacy whole-dive MD5
    digest lacked."""
    fs = _fs(
        images=[_image(f"{FOLDER}/A.ORF", "aa"), _image(f"{FOLDER}/B.ORF", "bb")],
        lookup={
            "aa": [{"image_id": 1, "dive_id": 64, "is_canonical": True}],
            "bb": [],
        },
    )

    report = await _finalize(412, _request(), _totals(), fs, monkeypatch)

    assert len(report.duplicate_overlap) == 1
    overlap = report.duplicate_overlap[0]
    assert overlap.dive_id == 64
    assert overlap.shared_images == 1
    assert overlap.containment == pytest.approx(0.5)


async def test_overlap_ignores_the_dive_being_ingested(monkeypatch):
    """The dive's own rows are in the lookup by construction — counting them
    would report containment 1.0 against itself on every single ingest."""
    fs = _fs(
        images=[_image(f"{FOLDER}/A.ORF", "aa")],
        lookup={"aa": [{"image_id": 1, "dive_id": 412, "is_canonical": True}]},
    )

    report = await _finalize(412, _request(), _totals(total=1, registered=1), fs, monkeypatch)

    assert report.duplicate_overlap == []


async def test_a_dive_with_no_overlap_reports_none(monkeypatch):
    fs = _fs(
        images=[_image(f"{FOLDER}/A.ORF", "aa")],
        lookup={"aa": []},
    )

    report = await _finalize(412, _request(), _totals(total=1, registered=1), fs, monkeypatch)

    assert report.duplicate_overlap == []
