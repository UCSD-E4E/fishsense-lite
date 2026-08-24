"""Unit tests for `preflight_ingest_activity` — the gate ingest has to pass.

Preflight writes nothing. Its whole job is to decide whether the folder can
become a dive, and to say so completely: **every problem at once, never
first-wins.** An operator submitting a folder from a boat has one round trip
worth of attention, and "fix this, resubmit, discover the next thing" spends it
badly.

The checks it makes are the ones nothing downstream can recover from:

  * A camera that can't be resolved, or has no intrinsics, silently gives
    stage 14 the wrong geometry — measurements come out plausible and wrong.
  * A folder spanning two serials is two rigs in one dive, which the schema
    has no way to express (spider only wrote a report file and carried on).
  * Unstated calibration intent produces a dive that can never be measured and
    never says why.
  * A frame with no readable timestamp corrupts stage-1 clustering, which is
    pure timestamp maths.
"""

from __future__ import annotations

from datetime import datetime, timezone
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

SERIAL = "BJ6C67989"
ROOT = "/fishsense_data/REEF/data"
FOLDER = f"{ROOT}/2024.06.20.REEF/082929_FishModels_FSL07"


def _camera(camera_id=7, serial=SERIAL, name="FSL-07"):
    from fishsense_api_sdk.models.camera import (
        Camera,
    )

    return Camera(id=camera_id, serial_number=serial, name=name)


def _entry(name: str, size: int = 15_000_000):
    from fishsense_api_workflow_worker.nas import (
        NasEntry,
    )

    return NasEntry(
        path=f"{FOLDER}/{name}", name=name, is_dir=False, size=size
    )


def _listing(names=("PA010001.ORF", "PA010002.ORF"), subfolders=()):
    from fishsense_api_workflow_worker.activities.list_dive_folder_activity import (
        DiveFolderListing,
    )

    return DiveFolderListing(
        folder_path=FOLDER,
        files=[_entry(n) for n in names],
        subfolders=list(subfolders),
    )


def _request(**kwargs):
    from fishsense_shared.ingest_contracts import (
        IngestDiveRequest,
    )

    kwargs.setdefault("dive_path", "2024.06.20.REEF/082929_FishModels_FSL07")
    kwargs.setdefault("self_calibrates", True)
    return IngestDiveRequest(**kwargs)


def _fs(cameras=None, intrinsics=object(), dives=None):
    """Fake SDK client: cameras to resolve against, intrinsics presence, and
    the existing dives that layer-1 duplicate detection compares leaf names to.
    """
    fs = MagicMock()
    fs.cameras.get = AsyncMock(
        return_value=[_camera()] if cameras is None else cameras
    )
    fs.cameras.get_intrinsics = AsyncMock(return_value=intrinsics)
    fs.dives.get = AsyncMock(return_value=dives or [])
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=False)
    return fs


async def _run(request, listing, monkeypatch, *, fs=None, headers=None):
    """Drive the activity with canned NAS header bytes, one per listed file.

    `headers` maps a file name to the bytes its ranged read returns; anything
    unlisted gets a well-formed default header.
    """
    from fishsense_api_workflow_worker.activities import (
        preflight_ingest_activity as sut,
    )

    default = build_orf(
        date_time="2024:08:21 08:56:51", serial_number=SERIAL, artist="FSL-07"
    )
    headers = headers or {}

    nas = MagicMock()
    nas.download_range.side_effect = lambda *, file_path, offset=0, length=0: (
        headers.get(file_path.rsplit("/", 1)[-1], default)
    )

    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs or _fs())
    # Preflight heartbeats per frame — a 500-frame dive is a long activity —
    # so it needs a real activity context.
    return await ActivityEnvironment().run(
        sut.preflight_ingest_activity, request, listing
    )


# ── the all-at-once contract ──────────────────────────────────────────


async def test_reports_every_error_at_once_rather_than_the_first(monkeypatch):
    """The property the whole activity exists for. A folder with three
    *independent* problems must come back with three errors, not one.

    Independence is the point of the fixture: the frames carry a good serial so
    the camera still resolves, which is what lets the missing-intrinsics check
    run at all. Faults that mask each other (an unreadable serial makes
    "has intrinsics?" unanswerable) would prove nothing about accumulation.
    """
    long_name = "P" + "A" * 300 + ".ORF"
    listing = _listing(names=("PA010001.ORF", long_name))

    preflight = await _run(
        _request(self_calibrates=False),  # no calibration intent
        listing,
        monkeypatch,
        fs=_fs(intrinsics=None),  # resolvable camera, but no intrinsics
    )

    joined = " | ".join(preflight.errors)
    assert "intrinsics" in joined
    assert "calibration" in joined
    assert "255" in joined
    assert len(preflight.errors) == 3


# ── camera resolution ─────────────────────────────────────────────────


async def test_resolves_the_camera_from_the_makernote_serial(monkeypatch):
    preflight = await _run(_request(), _listing(), monkeypatch)

    assert preflight.errors == []
    assert preflight.resolved_camera_id == 7
    assert preflight.resolved_camera_name == "FSL-07"


async def test_an_unknown_serial_fails_and_does_not_fall_back_to_artist(
    monkeypatch,
):
    """The anti-regression test for the camera decision. `Artist` is a free-text
    rig label; matching on it would resolve a camera whose intrinsics belong to
    different glass, and stage 14 would report confident wrong lengths. Better
    to refuse and make someone add the Camera row."""
    fs = _fs(cameras=[_camera(camera_id=3, serial="OTHER123", name="FSL-07")])

    preflight = await _run(_request(), _listing(), monkeypatch, fs=fs)

    assert preflight.resolved_camera_id is None
    assert any(SERIAL in e for e in preflight.errors)


async def test_an_explicit_camera_id_overrides_serial_resolution(monkeypatch):
    """The escape hatch for a body whose MakerNote is unreadable, or frames
    copied through a tool that stripped it."""
    fs = _fs(cameras=[_camera(camera_id=3, serial="OTHER123", name="FSL-03")])

    preflight = await _run(
        _request(camera_id=3), _listing(), monkeypatch, fs=fs
    )

    assert preflight.errors == []
    assert preflight.resolved_camera_id == 3


async def test_a_dive_spanning_two_serials_fails(monkeypatch):
    """One folder is one rig. Mixed intrinsics inside a single dive can't be
    expressed in the schema, and spider's response — write a report file and
    carry on — is how such dives got in unnoticed."""
    other = build_orf(
        date_time="2024:08:21 08:56:51",
        serial_number="XX9Z11111",
        artist="FSL-03",
    )

    preflight = await _run(
        _request(),
        _listing(),
        monkeypatch,
        headers={"PA010002.ORF": other},
    )

    assert any("serial" in e.lower() for e in preflight.errors)
    assert SERIAL in " ".join(preflight.errors)
    assert "XX9Z11111" in " ".join(preflight.errors)


async def test_a_camera_without_intrinsics_fails(monkeypatch):
    """Stage 14 needs intrinsics to exist before the dive is worth ingesting;
    discovering it later means a dive that sits in the cohort forever."""
    preflight = await _run(
        _request(), _listing(), monkeypatch, fs=_fs(intrinsics=None)
    )

    assert any("intrinsics" in e for e in preflight.errors)


async def test_artist_disagreeing_with_the_resolved_camera_is_a_warning(
    monkeypatch,
):
    """Not fatal — the serial is authoritative — but it means a mislabelled
    `Camera.name` or a re-housed body, and nothing else in the pipeline would
    ever notice."""
    fs = _fs(cameras=[_camera(camera_id=7, serial=SERIAL, name="FSL-99")])

    preflight = await _run(_request(), _listing(), monkeypatch, fs=fs)

    assert preflight.errors == []
    assert any("FSL-07" in w and "FSL-99" in w for w in preflight.warnings)


# ── calibration intent ────────────────────────────────────────────────


async def test_neither_calibration_intent_given_fails(monkeypatch):
    """A fish-only dive with no slate frames can never self-calibrate, so
    stage 14 can never measure it — and that is invisible in the files. The
    intent has to be stated at ingest or the dive quietly never completes."""
    preflight = await _run(
        _request(self_calibrates=False), _listing(), monkeypatch
    )

    assert any("calibration" in e for e in preflight.errors)


async def test_both_calibration_intents_given_fails(monkeypatch):
    """Contradictory intent. `own wins` would silently ignore the link the
    operator went to the trouble of specifying."""
    preflight = await _run(
        _request(self_calibrates=True, calibration_dive_id=61),
        _listing(),
        monkeypatch,
    )

    assert any("calibration" in e for e in preflight.errors)


async def test_borrowing_calibration_alone_is_valid(monkeypatch):
    preflight = await _run(
        _request(self_calibrates=False, calibration_dive_id=61),
        _listing(),
        monkeypatch,
    )

    assert preflight.errors == []


# ── per-frame validation ──────────────────────────────────────────────


async def test_a_frame_without_a_readable_timestamp_fails(monkeypatch):
    """Stage-1 clustering is pure timestamp maths, so a defaulted timestamp
    corrupts it silently. Rejecting the frame is the only safe answer."""
    blind = build_orf(
        date_time=None, date_time_original=None, serial_number=SERIAL
    )

    preflight = await _run(
        _request(), _listing(), monkeypatch, headers={"PA010002.ORF": blind}
    )

    assert any("PA010002.ORF" in e for e in preflight.errors)


async def test_the_timestamp_is_the_naive_exif_value_stamped_utc(monkeypatch):
    """The convention ~111k existing rows follow: the camera's wall clock,
    labelled UTC, with the recorded offset deliberately NOT applied. Matching
    it matters more than being right — mixing conventions inside one table is
    worse than a consistent offset."""
    preflight = await _run(_request(), _listing(), monkeypatch)

    assert preflight.images[0].taken_datetime == datetime(
        2024, 8, 21, 8, 56, 51, tzinfo=timezone.utc
    )


async def test_the_camera_offset_is_surfaced_but_not_applied(monkeypatch):
    with_offset = build_orf(
        date_time="2024:08:21 08:56:51",
        offset_time="-08:00",
        serial_number=SERIAL,
    )

    preflight = await _run(
        _request(),
        _listing(names=("PA010001.ORF",)),
        monkeypatch,
        headers={"PA010001.ORF": with_offset},
    )

    assert preflight.images[0].exif_offset == "-08:00"
    assert preflight.images[0].taken_datetime.hour == 8


async def test_a_fallback_timestamp_tag_is_warned_about(monkeypatch):
    """0x0132 missing means this body isn't the one the convention was derived
    from. The frame is still usable; the divergence should be visible."""
    fallback = build_orf(
        date_time=None,
        date_time_original="2024:08:21 08:56:51",
        serial_number=SERIAL,
    )

    preflight = await _run(
        _request(),
        _listing(names=("PA010001.ORF",)),
        monkeypatch,
        headers={"PA010001.ORF": fallback},
    )

    assert preflight.errors == []
    assert any("PA010001.ORF" in w for w in preflight.warnings)


async def test_a_path_over_255_characters_fails_and_names_the_offender(
    monkeypatch,
):
    """`Image.path` is varchar(255). Without this the dive half-ingests and
    fails on one row with a 422 that doesn't say which file."""
    long_name = "P" + "A" * 300 + ".ORF"

    preflight = await _run(
        _request(), _listing(names=("PA010001.ORF", long_name)), monkeypatch
    )

    assert any(long_name in e for e in preflight.errors)


async def test_an_empty_folder_fails(monkeypatch):
    """Almost always a mistyped path. Creating an empty dive would leave a row
    that no stage can ever act on."""
    preflight = await _run(_request(), _listing(names=()), monkeypatch)

    assert any("No .ORF frames" in e for e in preflight.errors)


# ── reporting ─────────────────────────────────────────────────────────


async def test_reads_only_the_first_megabyte_of_each_frame(monkeypatch):
    """What makes a dry run affordable: ~1 MB per file instead of ~15 MB — a
    500-frame dive previews for 0.5 GB rather than 7.5 GB. The saving is in
    bytes moved, so it holds whether the client resolves the read over SMB or
    falls back to FileStation."""
    from fishsense_api_workflow_worker.activities import (
        preflight_ingest_activity as sut,
    )

    nas = MagicMock()
    nas.download_range.return_value = build_orf(
        date_time="2024:08:21 08:56:51", serial_number=SERIAL
    )
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    monkeypatch.setattr(sut, "get_fs_client", _fs)

    await ActivityEnvironment().run(
        sut.preflight_ingest_activity, _request(), _listing()
    )

    for call in nas.download_range.call_args_list:
        assert call.kwargs["offset"] == 0
        assert call.kwargs["length"] == sut.EXIF_HEADER_BYTES


async def test_totals_and_subfolders_are_carried_into_the_report(monkeypatch):
    from fishsense_shared.ingest_contracts import (
        SubfolderReport,
    )

    listing = _listing(
        subfolders=[SubfolderReport(path=f"{FOLDER}/rollover", orf_count=47)]
    )

    preflight = await _run(_request(), listing, monkeypatch)

    assert preflight.total_bytes == 30_000_000
    assert preflight.subfolders[0].orf_count == 47
    assert any("rollover" in w for w in preflight.warnings)


async def test_a_leaf_name_collision_with_an_existing_dive_warns(monkeypatch):
    """Layer 1 of duplicate detection — free, since preflight already has the
    dive list. Catches the real prod case: dives 64 and 66 are both
    `082929_FishModels_FSL07`. Content-based containment needs checksums, so it
    can only run after the scan."""
    existing = MagicMock()
    existing.id = 64
    existing.path = "2023.01.01.REEF/082929_FishModels_FSL07"

    preflight = await _run(
        _request(), _listing(), monkeypatch, fs=_fs(dives=[existing])
    )

    assert preflight.errors == []
    assert any("64" in w for w in preflight.warnings)
