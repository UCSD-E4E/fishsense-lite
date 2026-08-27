# pylint: disable=C0121
"""Dive CRUD for FishSense API.

The stage-cohort selectors that used to live here are now in
`dive_cohort_controller` — they were ~600 lines with a different job, and
together the two modules had pushed this file past pylint's 1000-line ceiling.
"""

import logging
from datetime import datetime, timezone
from typing import List

from fastapi import Body, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.camera import Camera
from fishsense_api.models.camera_intrinsics import CameraIntrinsics
from fishsense_api.models.dive import Dive
from fishsense_api.models.dive_laser_line import DiveLaserLine
from fishsense_api.models.dive_slate import DiveSlate
from fishsense_api.models.image import Image
from fishsense_api.models.laser_extrinsics import LaserExtrinsics
from fishsense_api.models.priority import Priority
from fishsense_api.server import app

logger = logging.getLogger(__name__)


@app.get("/api/v1/dives/")
async def get_dives(session: AsyncSession = Depends(get_async_session)) -> List[Dive]:
    """Retrieve all dives."""
    logger.debug("Retrieving all dives")
    query = select(Dive)

    return (await session.exec(query)).all()


@app.get("/api/v1/canonical/dives/")
async def get_canonical_dives(
    session: AsyncSession = Depends(get_async_session),
) -> List[Dive]:
    """Retrieve all canonical dives."""
    logger.debug("Retrieving all canonical dives")
    query = (
        select(Dive)
        .distinct(Dive.id)
        .join_from(Dive, Image, Dive.id == Image.dive_id)
        .where(Image.is_canonical == True)
    )

    result = await session.exec(query)
    return result.all()


# Maximum length of `Dive.path` / `Image.path`, mirroring `max_length=255` on
# the models. Postgres would reject an over-long value, but sqlite silently
# stores it, so the check is explicit rather than left to the driver.
MAX_PATH_LENGTH = 255


@app.post("/api/v1/dives/", status_code=201)
async def post_dive(
    dive: Dive,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    # pylint: disable=too-many-branches
    # Each branch is one independent guard on a distinct field. Collapsing them
    # into a helper would hide which check produced which 422 detail, and the
    # detail strings are the point -- every one of these failures is otherwise
    # silent in a *later* pipeline stage.
    """Create or update a dive, keyed on its NAS-relative `path`.

    **Upserts on `path`.** `session.merge` keys on the primary key alone, so a
    body with `id=None` always INSERTs — which trips the unique index on
    `path` (the dive-shaped form of the #347 duplicate-key 500). Ingest
    re-posts the same path routinely: resuming a partial scan, and the
    finalize step re-POSTing to flip `priority` LOW -> HIGH. Resolving the
    natural key to an existing row id first turns the merge into an UPDATE.

    Validation is deliberately loud, because each of these breaks a *later*
    stage silently rather than here:

      * A dive whose camera has no `CameraIntrinsics` can never be measured by
        stage 14. Nothing errors — the dive just never reaches `measured`.
      * An over-long path is silently truncated by some backends, and the
        resulting `Image.path` no longer resolves on the NAS.
      * A self-referential or dangling `calibration_dive_id` makes
        `get_laser_extrinsics_for_dive`'s fallback either loop or 404.

    `priority` is NOT validated here: ingest deliberately creates dives at LOW
    and flips them to HIGH only once every image has landed, so LOW is a
    legitimate intermediate state (see the ingest workflow's commit flag).
    """
    logger.debug("Creating or updating dive with path=%s", dive.path)

    # Path checks run BEFORE `model_validate`. SQLModel's `max_length=255`
    # already rejects an over-long path for any normally-constructed body (so
    # an HTTP caller gets a 422 from request validation), but a body built via
    # `model_construct` bypasses that — and re-validating here would raise a
    # bare ValidationError instead of a useful 422.
    if not dive.path:
        raise HTTPException(status_code=422, detail="Dive path is required")
    if len(dive.path) > MAX_PATH_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Dive path exceeds {MAX_PATH_LENGTH} characters "
                f"({len(dive.path)}): {dive.path}"
            ),
        )

    # Which fields did the caller actually send? Captured BEFORE
    # `model_validate`, which rebuilds the object and marks every field as set.
    provided = set(dive.model_fields_set)

    # SQLModel `table=True` models skip pydantic coercion, so FastAPI hands us a
    # `Dive` whose `priority` is still the raw JSON `str`. Two consequences, both
    # fixed by coercing here rather than after `model_validate`:
    #   * an unrecognised value reaches `model_validate` and surfaces as a bare
    #     ValidationError -- a 500 where the caller deserves a 422;
    #   * `jsonable_encoder` below re-serializes the uncoerced object and emits a
    #     pydantic serializer warning on every single request.
    if dive.priority is not None and not isinstance(dive.priority, Priority):
        try:
            dive.priority = Priority(dive.priority)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail=f"Unknown priority {dive.priority!r}"
            ) from exc

    if dive.dive_datetime is None:
        # `Dive.dive_datetime` is annotated non-optional but carries
        # `default=None`, so FastAPI's request validation happily accepts a body
        # that omits it -- and the `model_validate` below then raises a bare
        # ValidationError, which surfaces as a 500. Guard explicitly.
        raise HTTPException(status_code=422, detail="dive_datetime is required")

    dive = Dive.model_validate(jsonable_encoder(dive))

    # Natural-key upsert on `path`.
    if dive.id is None:
        existing = (
            await session.exec(select(Dive).where(Dive.path == dive.path))
        ).first()
        if existing is not None:
            dive.id = existing.id
            # PARTIAL update. `session.merge` replaces every column of the
            # target row, so merging a body that only mentioned `priority`
            # would null `name`, `dive_slate_id` and `calibration_dive_id`.
            # That is data loss: `dive_slate_id` is written only by the
            # species-label sync (nulling it discards labeler work and stops
            # stages 9/12/13), and `calibration_dive_id` is the
            # borrowed-calibration link (nulling it stops stage 14 measuring
            # the dive). Neither failure reports anything.
            #
            # So overlay only what was sent onto the row as it stands. An
            # explicit `None` still clears a field -- it is in `provided`.
            # `model_dump()` (python mode), NOT `jsonable_encoder`: the latter
            # stringifies `priority`, and a table model doesn't coerce it back,
            # so the row would end up holding a raw str in an Enum column.
            merged = existing.model_dump()
            for field in provided:
                merged[field] = getattr(dive, field)
            merged["id"] = existing.id
            dive = Dive(**merged)

    # Validation runs on the EFFECTIVE row, after the overlay -- a partial
    # re-post that omits `camera_id` inherits the existing one and must not be
    # rejected for a field it never intended to change.
    if dive.camera_id is None:
        raise HTTPException(
            status_code=422,
            detail="camera_id is required; without it stage 14 cannot measure the dive",
        )
    camera = (
        await session.exec(select(Camera).where(Camera.id == dive.camera_id))
    ).first()
    if camera is None:
        raise HTTPException(
            status_code=422, detail=f"Camera {dive.camera_id} does not exist"
        )
    intrinsics = (
        await session.exec(
            select(CameraIntrinsics).where(CameraIntrinsics.camera_id == dive.camera_id)
        )
    ).first()
    if intrinsics is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Camera {dive.camera_id} has no intrinsics; stage 14 could never "
                "measure this dive"
            ),
        )

    if dive.calibration_dive_id is not None:
        if dive.calibration_dive_id == dive.id:
            raise HTTPException(
                status_code=422, detail="A dive cannot borrow its own calibration"
            )
        source = (
            await session.exec(select(Dive).where(Dive.id == dive.calibration_dive_id))
        ).first()
        if source is None:
            raise HTTPException(
                status_code=422,
                detail=f"Calibration source dive {dive.calibration_dive_id} does not exist",
            )

    dive = await session.merge(dive)
    await session.flush()

    dive_id = dive.id

    return dive_id


@app.get("/api/v1/dives/{dive_id}")
async def get_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> Dive | None:
    """Retrieve a dive by its ID."""
    logger.debug("Retrieving dive with id=%d", dive_id)
    query = select(Dive).where(Dive.id == dive_id)

    dive = (await session.exec(query)).first()
    if dive is None:
        logger.warning("Dive with id=%d not found", dive_id)
        raise HTTPException(status_code=404, detail="Dive not found")
    return dive


def _latest_extrinsics_query(dive_id: int):
    """The most-recently-created `LaserExtrinsics` row for a dive."""
    return (
        select(LaserExtrinsics)
        .where(LaserExtrinsics.dive_id == dive_id)
        .where(
            LaserExtrinsics.created_at
            == select(LaserExtrinsics.created_at)
            .where(LaserExtrinsics.dive_id == dive_id)
            .order_by(LaserExtrinsics.created_at.desc())  # pylint: disable=no-member
            .limit(1)
            .scalar_subquery()
        )
    )


@app.get("/api/v1/dives/{dive_id}/laser-extrinsics/")
async def get_laser_extrinsics_for_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> LaserExtrinsics | None:
    """Retrieve the laser extrinsics that apply to a dive.

    A dive's *own* calibration wins; if it has none but is linked to a
    calibration-source dive (`Dive.calibration_dive_id`), the source
    dive's extrinsics are returned instead. This lets a fish-only dive
    with no slate frames borrow the calibration of a sibling slate dive
    shot with the same camera+laser rig. The `laser_position` /
    `laser_axis` are all stage 14 consumes, so the returned row's
    `dive_id` (the source dive) is inconsequential to callers.
    """
    logger.debug("Retrieving laser extrinsics for dive with id=%d", dive_id)

    laser_extrinsics = (await session.exec(_latest_extrinsics_query(dive_id))).first()

    if laser_extrinsics is None:
        dive = await session.get(Dive, dive_id)
        if dive is not None and dive.calibration_dive_id is not None:
            logger.debug(
                "dive id=%d has no own extrinsics; borrowing from "
                "calibration_dive_id=%d",
                dive_id,
                dive.calibration_dive_id,
            )
            laser_extrinsics = (
                await session.exec(_latest_extrinsics_query(dive.calibration_dive_id))
            ).first()

    if laser_extrinsics is None:
        logger.warning("Laser extrinsics for dive with id=%d not found", dive_id)
        raise HTTPException(status_code=404, detail="Laser extrinsics not found")
    return laser_extrinsics


@app.put("/api/v1/dives/{dive_id}/laser-extrinsics/", status_code=201)
async def put_laser_extrinsics_for_dive(
    dive_id: int,
    extrinsics: LaserExtrinsics,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update laser extrinsics for a given dive ID.

    Upsert keyed on `dive_id`: a recompute overwrites the dive's existing row
    in place rather than appending a new one (calibration is per-dive and the
    table carries `uq_laserextrinsics_dive_id`). `created_at` is always stamped
    so the latest-wins read never sees a NULL timestamp — a NULL sorts first
    under `created_at DESC` in Postgres and made the lookup resolve nothing.
    """
    logger.debug("Creating or updating laser extrinsics for dive with id=%d", dive_id)
    extrinsics = LaserExtrinsics.model_validate(jsonable_encoder(extrinsics))
    extrinsics.dive_id = dive_id

    existing = (
        await session.exec(
            select(LaserExtrinsics).where(LaserExtrinsics.dive_id == dive_id)
        )
    ).first()
    if existing is not None:
        extrinsics.id = existing.id  # resolve natural key -> merge updates in place
    extrinsics.created_at = datetime.now(timezone.utc)

    extrinsics = await session.merge(extrinsics)
    await session.flush()

    extrinsics_id = extrinsics.id

    return extrinsics_id


@app.put("/api/v1/dives/{dive_id}/laser-line/", status_code=201)
async def put_dive_laser_line(
    dive_id: int,
    line: DiveLaserLine,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Upsert the per-dive laser-line fingerprint (keyed on `dive_id`).

    The laser-label validation refits this line each run; upserting keeps one
    row per dive so the fingerprint stays current (and tightens as outliers are
    superseded across runs). `fitted_at` is always stamped — same NULL-safe
    contract as `put_laser_extrinsics_for_dive`.
    """
    logger.debug("Upserting laser-line fingerprint for dive id=%d", dive_id)
    line = DiveLaserLine.model_validate(jsonable_encoder(line))
    line.dive_id = dive_id

    existing = (
        await session.exec(
            select(DiveLaserLine).where(DiveLaserLine.dive_id == dive_id)
        )
    ).first()
    if existing is not None:
        line.id = existing.id  # resolve natural key -> merge updates in place
    line.fitted_at = datetime.now(timezone.utc)

    line = await session.merge(line)
    await session.flush()

    return line.id


@app.get("/api/v1/dives/{dive_id}/laser-line/")
async def get_dive_laser_line(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> DiveLaserLine | None:
    """Return the dive's laser-line fingerprint, or None if not yet fitted."""
    return (
        await session.exec(
            select(DiveLaserLine).where(DiveLaserLine.dive_id == dive_id)
        )
    ).first()


@app.put("/api/v1/dives/{dive_id}/calibration-source/{source_dive_id}")
async def set_dive_calibration_source(
    dive_id: int,
    source_dive_id: int,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Link `dive_id` to borrow `source_dive_id`'s laser calibration.

    For a fish-only dive with no slate frames of its own: point it at a
    sibling slate/calibration dive shot with the same camera+laser rig.
    Laser-extrinsics resolution and the `calibrated` gate then fall back
    to the source dive when this dive has no calibration of its own.

    Returns the linked dive's id. 404 if either dive is missing; 400 on a
    self-link (a dive is never its own calibration source).
    """
    logger.debug(
        "Linking dive id=%d to calibration source dive id=%d",
        dive_id,
        source_dive_id,
    )
    if dive_id == source_dive_id:
        raise HTTPException(
            status_code=400,
            detail="A dive cannot be its own calibration source",
        )

    dive = await session.get(Dive, dive_id)
    if dive is None:
        raise HTTPException(status_code=404, detail="Dive not found")

    source = await session.get(Dive, source_dive_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Calibration source dive not found")

    dive.calibration_dive_id = source_dive_id
    session.add(dive)
    await session.flush()

    return dive_id


@app.delete("/api/v1/dives/{dive_id}/calibration-source/", status_code=204)
async def clear_dive_calibration_source(
    dive_id: int,
    session: AsyncSession = Depends(get_async_session),
) -> None:
    """Unlink `dive_id` from any borrowed calibration source (idempotent).

    404 only if the dive itself is missing; clearing an already-null link
    is a no-op.
    """
    logger.debug("Clearing calibration source link for dive id=%d", dive_id)
    dive = await session.get(Dive, dive_id)
    if dive is None:
        raise HTTPException(status_code=404, detail="Dive not found")

    dive.calibration_dive_id = None
    session.add(dive)
    await session.flush()


@app.put("/api/v1/dives/{dive_id}/dive-slate/{dive_slate_id}")
async def set_dive_slate(
    dive_id: int,
    dive_slate_id: int,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Set which `DiveSlate` template a dive was shot with.

    Identifies the physical slate (H-Slate / V-Slate N / Tic-Tac-Toe N),
    which stages 9/12/13 need before a dive can be slate-labeled and
    calibrated. Populated by the species-label sync from the labeler's
    slate-type choice; also settable by an operator.

    Returns the dive id. 404 if the dive or the DiveSlate template is
    missing.
    """
    logger.debug("Setting dive id=%d dive_slate_id=%d", dive_id, dive_slate_id)
    dive = await session.get(Dive, dive_id)
    if dive is None:
        raise HTTPException(status_code=404, detail="Dive not found")

    slate = await session.get(DiveSlate, dive_slate_id)
    if slate is None:
        raise HTTPException(status_code=404, detail="DiveSlate template not found")

    dive.dive_slate_id = dive_slate_id
    session.add(dive)
    await session.flush()

    return dive_id


@app.put("/api/v1/dives/{dive_id}/notes")
async def set_notes(
    dive_id: int,
    notes: str | None = Body(default=None, embed=True),
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Set (or clear) a dive's free-text operator note.

    A dedicated endpoint rather than a `post_dive` partial re-post because
    the species-label sync writes this automatically: the upsert path would
    have to read the row, overlay, and write it back, racing any concurrent
    write to the same dive. This touches exactly one column.

    Deliberately does NOT touch `priority`. The sync's caller knows a slate
    could not be identified, which is *evidence* a dive may never calibrate,
    but a dive can still borrow a sibling's calibration via
    `calibration_dive_id` — so parking it (`Priority.NONE`) stays a human
    decision informed by this note, not an automatic consequence of it.

    Returns the dive id. 404 if the dive is missing.
    """
    logger.debug("Setting dive id=%d notes", dive_id)
    dive = await session.get(Dive, dive_id)
    if dive is None:
        raise HTTPException(status_code=404, detail="Dive not found")

    dive.notes = notes
    session.add(dive)
    await session.flush()

    return dive_id
