"""Activity: decide whether a dive folder can be ingested, and write nothing.

Preflight is the gate. It reads each frame's EXIF header, resolves the camera,
runs every validation, and returns an `IngestPreflight`. A non-empty `errors`
list means the workflow refuses to proceed; `dry_run` returns this and stops.

**Every problem at once, never first-wins.** An operator submitting a folder
has one round trip worth of attention, and "fix this, resubmit, discover the
next thing" spends it badly. So the checks accumulate into a list rather than
raising at the first failure.

What it refuses, and why each is unrecoverable later:

  * **Unresolvable camera, and no `Artist` fallback.** `Artist` is a free-text
    rig label; matching on it would bind intrinsics belonging to different
    glass, and stage 14 would report confident wrong lengths. Refusing forces
    someone to add the `Camera` row, which is a minute's work and correct.
  * **A folder spanning two serials.** One folder is one rig; the schema has no
    way to say otherwise. Spider only wrote a report file and carried on, which
    is how such dives got in unnoticed.
  * **A camera with no intrinsics.** Discovered later, it means a dive that
    sits in the stage-14 cohort forever without ever producing a measurement.
  * **Unstated calibration intent.** A fish-only dive with no slate frames can
    never self-calibrate, so stage 14 can never measure it — and nothing in the
    files reveals that. Exactly one of `self_calibrates` / `calibration_dive_id`
    is required.
  * **A frame with no readable timestamp.** Stage-1 clustering is pure
    timestamp maths, so a defaulted value corrupts it silently.
  * **A path over 255 characters.** `Image.path` is varchar(255); without the
    check the dive half-ingests and fails on one row with a 422 that doesn't
    say which file.

Reads are ranged — the first megabyte holds the EXIF — which is what makes a
dry run affordable: ~1 MB per frame instead of ~15 MB. That saving is about
bytes moved, so it holds regardless of transport: `synology-filestation` >=0.2.0
prefers SMB and falls back to FileStation, and paces either way (`throttle=True`,
`max_concurrency=4`, `min_interval_ms=150`).

Reads are also serial here. Not because the transport can't cope — the client
handles its own pacing — but because a preflight shares the NAS with the hourly
staging activities doing real pipeline work, and a dry run should never be the
reason one of those is slow.

Duplicate detection here is **layer 1 only** — a leaf-name collision against
existing dives, free because the dive list is already fetched. Content-based
containment (layer 2) needs every frame's checksum, which needs the full files,
so it runs after the scan and lands in the final report.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List

from synology_filestation import DSMError
from temporalio import activity

from fishsense_api_workflow_worker.activities.list_dive_folder_activity import (
    DiveFolderListing,
)
from fishsense_api_workflow_worker.activities.nas_errors import (
    raise_if_permanent_dsm_error,
)
from fishsense_api_workflow_worker.activities.nas_frames import (
    build_nas_client,
)
from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_api_workflow_worker.config import settings
from fishsense_api_workflow_worker.exif import read_exif
from fishsense_shared.ingest_contracts import (
    IngestDiveRequest,
    IngestPreflight,
    PreflightImage,
)

# EXIF lives in the first megabyte of an ORF. Sized generously rather than
# tightly — the MakerNote's Equipment IFD sits well past the main IFD, and a
# short read costs a whole re-listing to discover.
EXIF_HEADER_BYTES = 1024 * 1024

# `Image.path` / `Dive.path` are varchar(255) (MAX_PATH_LENGTH in the API's
# image_controller / dive_controller). Checked here against the *stored*,
# share-relative form, not the absolute NAS path.
MAX_PATH_LENGTH = 255

__all__ = ["EXIF_HEADER_BYTES", "MAX_PATH_LENGTH", "preflight_ingest_activity"]


def _relative_path(absolute_path: str) -> str:
    """Strip `e4e_nas.raw_root_path` back off, giving the form the DB stores.

    Every existing `Image.path` is share-relative, and the staging activity
    re-prepends the root. Storing an absolute path would work until someone
    moved the share.

    A path **outside** the root keeps its leading slash, because that is the
    only form that survives the round trip. `resolve_nas_path` does exactly two
    things — prepend the root, or pass an absolute path through — so a relative
    path that is not under the root resolves to root + itself, a place that
    does not exist. FileStation reports that as a 502, which reads as a
    transient NAS problem rather than a malformed path, on every frame of the
    dive, forever. The 2025-01-17 pool test is the live case: same share,
    outside `REEF/data`.
    """
    root = settings.e4e_nas.raw_root_path.rstrip("/") + "/"
    if absolute_path.startswith(root):
        return absolute_path[len(root) :]
    return absolute_path


def _parse_taken_datetime(raw: str | None) -> datetime | None:
    """`"YYYY:MM:DD HH:MM:SS"` → an aware UTC datetime.

    The value is the camera's wall clock, and the recorded UTC offset is
    deliberately **not** applied — that is the convention ~111k existing rows
    follow. Matching it matters more than being right: a consistent offset is
    recoverable later, two conventions mixed inside one column is not.
    """
    if not raw:
        return None
    try:
        naive = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    return naive.replace(tzinfo=timezone.utc)


async def _read_header(nas, file_path: str) -> bytes:
    """One ranged read. No inner retry — Temporal's bounded policy owns
    backoff, and an inner loop under it is what tripped the NAS auto-block."""
    try:
        return await asyncio.to_thread(
            nas.download_range,
            file_path=file_path,
            offset=0,
            length=EXIF_HEADER_BYTES,
        )
    except DSMError as exc:
        raise_if_permanent_dsm_error(exc, context=file_path)
        raise


def _check_calibration_intent(request: IngestDiveRequest) -> List[str]:
    """Exactly one of the two must be given.

    Both is contradictory: `get_laser_extrinsics_for_dive` resolves own-wins,
    so the link the operator took trouble to specify would be silently ignored.
    Neither leaves a dive that can never be measured and never says why.
    """
    borrows = request.calibration_dive_id is not None
    if request.self_calibrates and borrows:
        return [
            "Contradictory calibration intent: self_calibrates=True and "
            f"calibration_dive_id={request.calibration_dive_id}. A dive with "
            "its own slate always self-calibrates, so the link would be "
            "ignored — pass exactly one."
        ]
    if not request.self_calibrates and not borrows:
        return [
            "No calibration intent given. Pass self_calibrates=True if this "
            "dive has its own slate frames, or calibration_dive_id=<dive> to "
            "borrow a sibling's calibration. Without one, stage 14 can never "
            "measure this dive."
        ]
    return []


async def _resolve_camera(fs, request, serials, artists, errors, warnings):
    """Resolve to a `(camera_id, camera_name)` pair, or `(None, None)`."""
    if len(serials) > 1:
        errors.append(
            "Frames span more than one camera serial "
            f"({', '.join(sorted(serials))}). One folder is one rig — split "
            "the folder and submit each dive separately."
        )
        return None, None

    cameras = await fs.cameras.get() or []

    if request.camera_id is not None:
        match = next((c for c in cameras if c.id == request.camera_id), None)
        if match is None:
            errors.append(f"No Camera row with id {request.camera_id}.")
            return None, None
        return match.id, match.name

    if not serials:
        errors.append(
            "No camera serial found in any frame's Olympus MakerNote, and no "
            "camera_id override was given."
        )
        return None, None

    serial = next(iter(serials))
    match = next((c for c in cameras if c.serial_number == serial), None)
    if match is None:
        errors.append(
            f"Camera serial {serial} matches no Camera row. Add the camera "
            "(with its intrinsics) before ingesting. Deliberately not falling "
            "back to the EXIF Artist tag — a free-text rig label would bind "
            "the wrong intrinsics and stage 14 would report confident wrong "
            "lengths."
        )
        return None, None

    # The serial is authoritative; a disagreeing Artist means a mislabelled
    # Camera.name or a re-housed body. Nothing else in the pipeline would ever
    # notice, so say it here.
    for artist in sorted(a for a in artists if a and a != match.name):
        warnings.append(
            f"EXIF Artist {artist!r} disagrees with the resolved camera's "
            f"name {match.name!r} (serial {serial})."
        )
    return match.id, match.name


@activity.defn
async def preflight_ingest_activity(
    request: IngestDiveRequest, listing: DiveFolderListing
) -> IngestPreflight:
    # pylint: disable=too-many-locals,too-many-branches
    # Preflight is a checklist: the branch count IS the feature. Splitting it
    # into per-check helpers would scatter the "collect, never raise" contract
    # that makes the all-at-once report work.
    errors: List[str] = []
    warnings: List[str] = []

    errors.extend(_check_calibration_intent(request))

    if not listing.files:
        errors.append(
            f"No .ORF frames directly inside {listing.folder_path}. Ingest is "
            "non-recursive — a dive is exactly one directory."
        )

    for subfolder in listing.subfolders:
        warnings.append(
            f"{subfolder.path} contains {subfolder.orf_count} .ORF files. "
            "Under the existing convention that is a separate dive — submit it "
            "as its own request; it is not included here."
        )

    nas = build_nas_client()
    images: List[PreflightImage] = []
    serials: set[str] = set()
    artists: set[str] = set()

    for entry in listing.files:
        activity.heartbeat(entry.path)
        stored_path = _relative_path(entry.path)
        if len(stored_path) > MAX_PATH_LENGTH:
            errors.append(
                f"Path exceeds {MAX_PATH_LENGTH} characters "
                f"({len(stored_path)}): {stored_path}"
            )
            continue

        exif = read_exif(await _read_header(nas, entry.path))
        taken = _parse_taken_datetime(exif.date_time)
        if taken is None:
            errors.append(
                f"No readable EXIF timestamp in {stored_path}. Stage-1 "
                "clustering is pure timestamp maths, so the frame cannot be "
                "ingested with a defaulted value."
            )
            continue
        if exif.date_time_is_fallback:
            warnings.append(
                f"{stored_path} has no DateTime (0x0132); fell back to "
                "DateTimeOriginal (0x9003)."
            )

        if exif.serial_number:
            serials.add(exif.serial_number)
        if exif.artist:
            artists.add(exif.artist)

        images.append(
            PreflightImage(
                path=stored_path,
                size=entry.size,
                taken_datetime=taken,
                exif_offset=exif.offset_time,
                serial_number=exif.serial_number,
                artist=exif.artist,
            )
        )

    async with get_fs_client() as fs:
        camera_id, camera_name = await _resolve_camera(
            fs, request, serials, artists, errors, warnings
        )

        if camera_id is not None:
            if await fs.cameras.get_intrinsics(camera_id) is None:
                errors.append(
                    f"Camera {camera_id} ({camera_name}) has no intrinsics. "
                    "Stage 14 cannot measure this dive until they exist."
                )

        # Layer 1 duplicate detection: leaf-name collision. Free, since the
        # dive list is already here. Catches the real prod case — dives 64 and
        # 66 are both `082929_FishModels_FSL07`. Content-based containment
        # needs checksums, so it runs after the scan.
        leaf = listing.folder_path.rstrip("/").rsplit("/", 1)[-1]
        for dive in await fs.dives.get() or []:
            if not dive.path:
                continue
            if dive.path.rstrip("/").rsplit("/", 1)[-1] == leaf:
                warnings.append(
                    f"Dive {dive.id} has the same folder name ({leaf!r}) at "
                    f"{dive.path}. Dive names are not unique in prod; this may "
                    "be a re-ingest."
                )

    activity.logger.info(
        "preflight path=%s frames=%d errors=%d warnings=%d",
        listing.folder_path,
        len(images),
        len(errors),
        len(warnings),
    )
    return IngestPreflight(
        dive_path=listing.folder_path,
        images=images,
        subfolders=list(listing.subfolders),
        resolved_camera_id=camera_id,
        resolved_camera_name=camera_name,
        total_bytes=sum(e.size for e in listing.files),
        errors=errors,
        warnings=warnings,
    )
