"""Unit tests for `verify_dive_checksums_activity` — does the migrated data
mean what we think it means?

Every ingest convention was recovered by reading the retired spider crawler's
source: checksum is `md5` of the whole file (`backend.py:67`), and
`taken_datetime` is naive EXIF tag 0x0132 stamped UTC (`backend.py:20`). That
derivation is solid, but it proves what *spider wrote*. It does not prove that
all ~131k rows came from spider, and no stored value has ever been compared
against the bytes still sitting on the NAS.

The gap matters because of how it fails. If existing checksums were computed
even slightly differently, duplicate detection does not error — it silently
reports **zero overlap**. A re-ingest of an already-present dive would look
entirely new, every frame would land `is_canonical=True`, and the canonical-only
gating would have nothing to gate on.

So this activity re-hashes real files and reports. It is **read-only**: no
writes to the NAS, no writes through the SDK. A mismatch is a *finding*, not a
failure — the run completes and reports everything it saw, because "which rows
disagree, and how" is the question being asked.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from temporalio.testing import ActivityEnvironment

# pylint: disable=import-error  # noqa: E501
# `tests` is not a unique package name in this workspace — both
# services/fishsense-api-workflow-worker/tests and libs/*/tests are called
# `tests`, and pylint binds the name to whichever it parses FIRST. CI lints
# `git diff --name-only` output, which is alphabetical, so a changed
# libs/*/tests file wins and this relative import stops resolving. The
# import itself is correct — pytest resolves it fine.
from ._tiff_builder import build_orf

ROOT = "/fishsense_data/REEF/data"
DIVE_FOLDER = "2024.06.20.REEF/082929_FishModels_FSL07"


def _orf(date_time="2024:08:21 08:56:51") -> bytes:
    """A synthetic ORF, padded past one hash chunk so the streaming read is
    exercised rather than incidentally fitting in a single buffer."""
    return build_orf(date_time=date_time, serial_number="BJ6C67989") + b"\0" * 40_000


def _image(image_id, name, data, *, checksum=None, taken=None):
    """An `Image` row as the SDK returns it, defaulting to values that agree
    with `data` so a test only has to state the disagreement it cares about."""
    image = MagicMock()
    image.id = image_id
    image.path = f"{DIVE_FOLDER}/{name}"
    image.checksum = (
        checksum if checksum is not None else hashlib.md5(data).hexdigest()
    )
    image.taken_datetime = (
        taken
        if taken is not None
        else datetime(2024, 8, 21, 8, 56, 51, tzinfo=timezone.utc)
    )
    image.is_canonical = True
    return image


def _fs(images):
    fs = MagicMock()
    fs.images.get = AsyncMock(return_value=images)
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=False)
    return fs


def _nas(contents: dict[str, bytes]):
    """A fake NAS that materialises real files, keyed by basename.

    Real files rather than in-memory bytes because the activity must hash from
    disk in chunks — a 15 MB buffer per frame times a dive is real memory for
    no benefit, and that is exactly what spider avoided.
    """
    nas = MagicMock()

    def _download_to(*, src_path: str, dest_dir: str):
        name = src_path.rsplit("/", 1)[-1]
        if name not in contents:
            from synology_filestation import (
                DSMError,
            )

            raise DSMError("Synology API error 408")
        Path(dest_dir, name).write_bytes(contents[name])

    nas.download_to.side_effect = _download_to
    return nas


async def _run(images, contents, monkeypatch, **kwargs):
    from fishsense_api_workflow_worker.activities import (
        verify_dive_checksums_activity as sut,
    )

    monkeypatch.setattr(sut, "_build_nas_client", lambda: _nas(contents))
    monkeypatch.setattr(sut, "get_fs_client", lambda: _fs(images))
    return await ActivityEnvironment().run(
        sut.verify_dive_checksums_activity, 412, kwargs.get("limit")
    )


# ── the question being asked ──────────────────────────────────────────


async def test_a_row_whose_stored_checksum_matches_the_file_is_counted_matched(
    monkeypatch,
):
    data = _orf()
    report = await _run(
        [_image(1, "PA010001.ORF", data)], {"PA010001.ORF": data}, monkeypatch
    )

    assert report.checked == 1
    assert report.checksum_matched == 1
    assert report.mismatches == []


async def test_a_checksum_mismatch_reports_both_values_and_the_path(monkeypatch):
    """The finding has to be actionable. "Some rows disagree" would send
    someone back to the NAS to work out which."""
    data = _orf()
    stored = "0" * 32
    report = await _run(
        [_image(1, "PA010001.ORF", data, checksum=stored)],
        {"PA010001.ORF": data},
        monkeypatch,
    )

    assert report.checksum_matched == 0
    assert len(report.mismatches) == 1
    finding = report.mismatches[0]
    assert finding.path.endswith("PA010001.ORF")
    assert finding.stored == stored
    assert finding.computed == hashlib.md5(data).hexdigest()


async def test_the_checksum_is_md5_of_the_whole_file(monkeypatch):
    """Pins the convention recovered from `spider/backend.py:67`. A reader that
    hashed only the header, or canonicalised anything, would agree with itself
    forever and disagree with every migrated row."""
    data = _orf()
    report = await _run(
        [_image(1, "PA010001.ORF", data, checksum=hashlib.md5(data).hexdigest())],
        {"PA010001.ORF": data},
        monkeypatch,
    )

    assert report.checksum_matched == 1


async def test_a_timestamp_mismatch_is_reported_separately(monkeypatch):
    """The second migrated assumption, free once the bytes are local. Tracked
    apart from checksums because the two have different consequences — a wrong
    checksum breaks duplicate detection, a wrong timestamp breaks stage-1
    clustering."""
    data = _orf()
    wrong = datetime(1999, 1, 1, tzinfo=timezone.utc)
    report = await _run(
        [_image(1, "PA010001.ORF", data, taken=wrong)],
        {"PA010001.ORF": data},
        monkeypatch,
    )

    assert report.checksum_matched == 1
    assert len(report.timestamp_mismatches) == 1
    assert report.timestamp_mismatches[0].stored == wrong.isoformat()


async def test_the_stored_timestamp_convention_is_naive_0x0132_stamped_utc(
    monkeypatch,
):
    """Agreement, not correctness. The camera's offset is deliberately not
    applied — matching the ~111k existing rows matters more than being right,
    because one consistent offset is recoverable and two mixed conventions in
    one column are not."""
    data = _orf(date_time="2024:08:21 08:56:51")
    report = await _run(
        [
            _image(
                1,
                "PA010001.ORF",
                data,
                taken=datetime(2024, 8, 21, 8, 56, 51, tzinfo=timezone.utc),
            )
        ],
        {"PA010001.ORF": data},
        monkeypatch,
    )

    assert report.timestamp_mismatches == []


# ── findings, not failures ────────────────────────────────────────────


async def test_a_file_missing_from_the_nas_is_a_finding_not_a_crash(monkeypatch):
    """The difference from the staging activity, which treats a missing file as
    a non-retryable failure. Here "the row exists but the file is gone" is one
    of the answers being looked for, so the run reports it and carries on."""
    data = _orf()
    report = await _run(
        [
            _image(1, "PA010001.ORF", data),
            _image(2, "GONE.ORF", data),
        ],
        {"PA010001.ORF": data},
        monkeypatch,
    )

    assert report.checked == 2
    assert report.checksum_matched == 1
    assert [m.path.rsplit("/", 1)[-1] for m in report.missing_on_nas] == [
        "GONE.ORF"
    ]


async def test_a_row_with_no_stored_checksum_is_reported_not_silently_skipped(
    monkeypatch,
):
    """A NULL checksum is itself a migration finding — the column is what
    duplicate detection joins on, so a blank one silently excludes the row from
    every overlap calculation."""
    data = _orf()
    image = _image(1, "PA010001.ORF", data)
    image.checksum = None  # `_image` derives one by default; this is the point

    report = await _run([image], {"PA010001.ORF": data}, monkeypatch)

    assert report.checked == 1
    assert report.checksum_matched == 0
    assert report.mismatches == []
    assert len(report.no_stored_checksum) == 1
    assert report.no_stored_checksum[0].computed == hashlib.md5(data).hexdigest()


async def test_one_bad_row_does_not_stop_the_rest_being_checked(monkeypatch):
    data = _orf()
    report = await _run(
        [
            _image(1, "A.ORF", data, checksum="0" * 32),
            _image(2, "B.ORF", data),
            _image(3, "C.ORF", data),
        ],
        {"A.ORF": data, "B.ORF": data, "C.ORF": data},
        monkeypatch,
    )

    assert report.checked == 3
    assert report.checksum_matched == 2
    assert len(report.mismatches) == 1


# ── cost control ──────────────────────────────────────────────────────


async def test_a_limit_caps_how_many_frames_are_downloaded(monkeypatch):
    """A dive is ~500 frames at ~15 MB. Verification pulls whole files — the
    one thing preflight's ranged read avoids — so a sample has to be possible.
    Answering "does the convention hold" does not need every frame."""
    data = _orf()
    images = [_image(i, f"P{i:04d}.ORF", data) for i in range(10)]
    contents = {f"P{i:04d}.ORF": data for i in range(10)}

    report = await _run(images, contents, monkeypatch, limit=3)

    assert report.checked == 3
    assert report.total_in_dive == 10


# ── read-only ─────────────────────────────────────────────────────────


def test_the_module_imports_no_write_capable_client():
    """Tripwire, mirroring the cleanup activity's. This runs against prod data
    to answer a question; it must not be able to change the answer. Catches a
    future edit that reaches for `upload`, `delete` or an SDK write.
    """
    import inspect

    from fishsense_api_workflow_worker.activities import (
        verify_dive_checksums_activity as sut,
    )

    source = inspect.getsource(sut)
    for forbidden in (".upload(", ".delete(", ".post(", ".put(", "upload_bytes"):
        assert forbidden not in source, f"verification must stay read-only: {forbidden}"
