# pylint: disable=C0121
"""Label Controller for FishSense API."""

import logging
from typing import List

from fastapi import Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlalchemy import alias
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.dive import Dive
from fishsense_api.models.dive_slate_label import DiveSlateLabel
from fishsense_api.models.head_tail_label import HeadTailLabel
from fishsense_api.models.image import Image
from fishsense_api.models.label_studio_sync_cursor import LabelStudioSyncCursor
from fishsense_api.models.laser_label import LaserLabel
from fishsense_api.models.laser_prediction import LaserPrediction
from fishsense_api.models.species_label import SpeciesLabel
from fishsense_api.server import app

logger = logging.getLogger(__name__)


async def _upsert_label(session: AsyncSession, model, image_id: int, payload):
    """Upsert a label without clearing fields the caller never mentioned.

    `session.merge` writes every column of the model it is handed, and FastAPI
    builds that model from the request body with **defaults for anything
    absent**. So a writer that constructs a label with the twelve fields it
    cares about silently wipes the ones it does not.

    Prod 2026-09-07: populate omits `needs_reprocess`, and its image set is
    exactly the one a reprocess flag marks, so a dive lost 259 flags within
    the hour -- before the render they asked for had started.

    **`LaserPrediction` shares the mechanism and must NOT share the fix.**
    CLAUDE.md records the same merge-clobber there -- re-predicting a dive
    clears its gate verdicts -- but records it as *correct*: a verdict computed
    from a dot the row no longer holds is stale, and clearing it is what drops
    the dive off the landing page until the gate has been back through it.
    Propagating this helper into `_prediction_upsert.py` would break that.

    **It does not cover read-modify-write callers, and cannot.** It protects a
    writer that names only the fields it sets; the hourly syncs fetch a label,
    mutate two fields and PUT the whole model back, so `needs_reprocess` really
    is in their body, carrying the value they read. A flag raised between that
    read and that write is still lost. Closing that means making the flag
    unwritable here -- it has its own PUT/DELETE routes -- a deliberate API
    change, not a bug fix.

    `payload.model_fields_set` holds the keys actually present in the request
    body, so it must be read **before** any re-validation: round-tripping
    through `jsonable_encoder` marks every field as set.
    """
    provided = set(payload.model_fields_set)
    payload = model.model_validate(jsonable_encoder(payload))
    payload.image_id = image_id
    provided.add("image_id")

    if payload.id is None:
        if "label_studio_project_id" in provided:
            # Natural-key upsert — see `_resolve_label_natural_key` for why
            # merge alone duplicates, including the NULL-project case.
            payload.id = await _resolve_label_natural_key(
                session, model, image_id, payload.label_studio_project_id
            )
        else:
            # An absent project id is not a null one. Both arrive here as
            # `None`, but only `provided` can tell them apart, and resolving
            # the natural key on the default would look for a project-less row,
            # miss the real one, and INSERT a second. That stray row is a
            # sentinel — `completed` false — so it pins
            # `dive_pipeline_status.*_labeling_complete` false for the dive
            # until someone deletes it by hand.
            payload.id = await _resolve_sole_label_for_image(session, model, image_id)

    if payload.id is not None:
        existing = await session.get(model, payload.id)
        if existing is not None:
            if existing.image_id != image_id:
                # The id came from the request body and names another image's
                # row. Writing it would move that row onto this image —
                # `image_id` is always in `provided` — so one image silently
                # loses its label and this one gains the other's coordinates.
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"label id={payload.id} belongs to image "
                        f"{existing.image_id}, not {image_id}"
                    ),
                )
            for name in provided:
                if name != "id":
                    setattr(existing, name, getattr(payload, name))
            session.add(existing)
            await session.flush()
            return existing.id

    merged = await session.merge(payload)
    await session.flush()
    return merged.id


async def _resolve_sole_label_for_image(
    session: AsyncSession, model, image_id: int
) -> int | None:
    """The image's only label id, None if it has none, 409 if it has several.

    Used when the body names no project, so the natural key cannot be formed.
    An image legitimately carries one label per Label Studio project, and with
    more than one there is no fact in the request saying which was meant —
    picking one would be a coin flip over whose data gets overwritten. Refusing
    is the only answer that cannot corrupt, and the caller can always say which
    by sending `label_studio_project_id`.
    """
    rows = (await session.exec(select(model).where(model.image_id == image_id))).all()
    if not rows:
        return None
    if len(rows) > 1:
        raise HTTPException(
            status_code=409,
            detail=(
                f"image {image_id} has {len(rows)} {model.__name__} rows; send "
                "label_studio_project_id to say which one this updates"
            ),
        )
    return rows[0].id


async def _resolve_label_natural_key(
    session: AsyncSession, model, image_id: int, project_id: int | None
) -> int | None:
    """Existing row id for `(image_id, project_id)`, or None.

    `session.merge` keys on the PRIMARY key alone, so a body with `id=None`
    always INSERTs. Every label model carries a `uq_<kind>_image_project`
    constraint on `(image_id, label_studio_project_id)`, so that second INSERT
    is a duplicate-key 500 — exactly what a populate retry produced (prod
    headtail dive 347). Resolving the natural key first turns the merge into an
    UPDATE. Same shape as `post_measurement` / `uq_measurement_image_fish`.

    The NULL-project branch is the one that is easy to miss. Sentinel rows
    carry `label_studio_project_id = NULL`, and SQL treats NULLs as DISTINCT in
    a unique constraint — `(11, NULL)` never conflicts with `(11, NULL)`. So
    the constraint cannot protect those rows, and skipping resolution for them
    (which the four handlers used to do) meant every write appended another
    row, unbounded. `IS NULL` has to be spelled explicitly because `== None`
    renders as `= NULL`, which is never true.

    Spelled once rather than four times: the handlers were four textual copies
    that `duplicate-code` cannot see, because the differing model name makes
    every line differ.
    """
    query = select(model).where(model.image_id == image_id)
    if project_id is None:
        query = query.where(model.label_studio_project_id.is_(None))
    else:
        query = query.where(model.label_studio_project_id == project_id)
    existing = (await session.exec(query)).first()
    return existing.id if existing is not None else None


@app.get("/api/v1/labels/dive-slate/label-studio-project-ids")
async def get_dive_slate_label_studio_project_ids(
    incomplete: bool = False,
    session: AsyncSession = Depends(get_async_session),
) -> List[int]:
    """Distinct Label Studio project IDs that have at least one dive-slate label.

    `incomplete=true` narrows to projects that have at least one label
    where `completed` is NULL or false. Backs the `apps/fishsense-lite-web/` SSR
    landing page, which surfaces only LS projects with outstanding
    labeling work.

    NOTE: must precede the `/api/v1/labels/dive-slate/{image_id}` route —
    Starlette's default path converter treats `{image_id}` as `[^/]+`,
    so registration order is what disambiguates the literal segment.
    """
    logger.debug(
        "Retrieving distinct Label Studio project IDs with dive-slate labels "
        "(incomplete=%s)",
        incomplete,
    )
    query = (
        select(DiveSlateLabel.label_studio_project_id).where(
            DiveSlateLabel.label_studio_project_id != None
        )
        # Exclude dead-lettered rows — parity with every other read.
        .where(DiveSlateLabel.superseded == False)
    )
    if incomplete:
        query = query.where(
            (DiveSlateLabel.completed == False)
            | (DiveSlateLabel.completed.is_(None))  # pylint: disable=no-member
        )
    return list((await session.exec(query.distinct())).all())


@app.get("/api/v1/labels/dive-slate/{image_id}")
async def get_dive_slate_label(
    image_id: int, session: AsyncSession = Depends(get_async_session)
) -> DiveSlateLabel | None:
    """Retrieve slate label for a given image ID."""
    logger.debug("Retrieving dive slate label for image with id=%d", image_id)

    query = (
        select(DiveSlateLabel)
        .where(DiveSlateLabel.image_id == image_id)
        .where(DiveSlateLabel.superseded == False)
    )

    return (await session.exec(query)).first()


@app.get("/api/v1/dives/{dive_id}/labels/dive-slate")
async def get_dive_slate_labels_for_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> List[DiveSlateLabel]:
    """Retrieve all slate labels for a given dive ID."""
    logger.debug("Retrieving dive slate labels for dive with id=%d", dive_id)
    query = (
        select(DiveSlateLabel)
        .join_from(DiveSlateLabel, Image, DiveSlateLabel.image_id == Image.id)
        .join_from(Image, Dive, Image.dive_id == Dive.id)
        .where(Dive.id == dive_id)
        .where(DiveSlateLabel.superseded == False)
        .order_by(DiveSlateLabel.image_id)  # see get_laser_labels_for_dive
    )

    labels = (await session.exec(query)).all()
    if not labels:
        logger.warning("Dive slate labels for dive with id=%d not found", dive_id)
        raise HTTPException(status_code=404, detail="Labels not found")
    return labels


@app.put("/api/v1/labels/dive-slate/{image_id}", status_code=201)
async def put_dive_slate_label(
    image_id: int,
    label: DiveSlateLabel,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update slate label for a given image ID."""
    logger.debug("Creating or updating dive slate label for image with id=%d", image_id)
    return await _upsert_label(session, DiveSlateLabel, image_id, label)


@app.get("/api/v1/labels/dive-slate/label-studio/{label_studio_id}")
async def get_dive_slate_label_by_label_studio_id(
    label_studio_id: int, session: AsyncSession = Depends(get_async_session)
) -> DiveSlateLabel | None:
    """Retrieve a dive-slate label for a given Label Studio task ID."""
    logger.debug("Retrieving dive-slate label for Label Studio id=%d", label_studio_id)
    query = (
        select(DiveSlateLabel)
        .where(DiveSlateLabel.label_studio_task_id == label_studio_id)
        .where(DiveSlateLabel.superseded == False)
    )

    label = (await session.exec(query)).first()
    if label is None:
        logger.warning(
            "Dive-slate label for Label Studio id=%d not found", label_studio_id
        )
        raise HTTPException(status_code=404, detail="Label not found")
    return label


@app.get("/api/v1/labels/headtail/label-studio-project-ids")
async def get_headtail_label_studio_project_ids(
    incomplete: bool = False,
    session: AsyncSession = Depends(get_async_session),
) -> List[int]:
    """Distinct Label Studio project IDs that have at least one head-tail label.

    `incomplete=true` narrows to projects that have at least one label
    where `completed` is NULL or false. Backs the `apps/fishsense-lite-web/` SSR
    landing page, which surfaces only LS projects with outstanding
    labeling work.

    Replaces a per-dive fan-out the api-workflow-worker used to do (one
    HTTP round trip per canonical dive) — that approach blew past the
    activity's 10-minute schedule_to_close timeout as the dataset grew.

    NOTE: must precede the `/api/v1/labels/headtail/{image_id}` route —
    Starlette's default path converter treats `{image_id}` as `[^/]+`,
    so registration order is what disambiguates the literal segment.
    """
    logger.debug(
        "Retrieving distinct Label Studio project IDs with head-tail labels "
        "(incomplete=%s)",
        incomplete,
    )
    query = (
        select(HeadTailLabel.label_studio_project_id).where(
            HeadTailLabel.label_studio_project_id != None
        )
        # Parity with laser + every other headtail read — exclude superseded.
        .where(HeadTailLabel.superseded == False)
    )
    if incomplete:
        query = query.where(
            (HeadTailLabel.completed == False)
            | (HeadTailLabel.completed.is_(None))  # pylint: disable=no-member
        )
    return list((await session.exec(query.distinct())).all())


@app.get("/api/v1/labels/headtail/{image_id}")
async def get_headtail_label(
    image_id: int, session: AsyncSession = Depends(get_async_session)
) -> HeadTailLabel | None:
    """Retrieve a head-tail label for a given image ID."""
    logger.debug("Retrieving head-tail label for image with id=%d", image_id)

    query = (
        select(HeadTailLabel)
        .where(HeadTailLabel.image_id == image_id)
        .where(HeadTailLabel.superseded == False)
    )

    label = (await session.exec(query)).first()
    if label is None:
        logger.warning("Head-tail label for image with id=%d not found", image_id)
        raise HTTPException(status_code=404, detail="Label not found")
    return label


@app.get("/api/v1/dives/{dive_id}/labels/headtail")
async def get_headtail_labels_for_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> List[HeadTailLabel]:
    """Retrieve all head-tail labels for a given dive ID."""
    logger.debug("Retrieving head-tail labels for dive with id=%d", dive_id)
    query = (
        select(HeadTailLabel)
        .join_from(HeadTailLabel, Image, HeadTailLabel.image_id == Image.id)
        .join_from(Image, Dive, Image.dive_id == Dive.id)
        .where(Dive.id == dive_id)
        .where(HeadTailLabel.superseded == False)
        .order_by(HeadTailLabel.image_id)  # see get_laser_labels_for_dive
    )

    labels = (await session.exec(query)).all()
    if not labels:
        logger.warning("Head-tail labels for dive with id=%d not found", dive_id)
        raise HTTPException(status_code=404, detail="Labels not found")
    return labels


@app.put("/api/v1/labels/headtail/{image_id}", status_code=201)
async def put_headtail_label(
    image_id: int,
    label: HeadTailLabel,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update a head-tail label for a given image ID."""
    logger.debug("Creating or updating headtail label for image with id=%d", image_id)
    return await _upsert_label(session, HeadTailLabel, image_id, label)


@app.get("/api/v1/labels/headtail/label-studio/{label_studio_id}")
async def get_headtail_label_by_label_studio_id(
    label_studio_id: int, session: AsyncSession = Depends(get_async_session)
) -> HeadTailLabel | None:
    """Retrieve a head-tail label for a given Label Studio ID."""
    logger.debug("Retrieving head-tail label for Label Studio id=%d", label_studio_id)
    query = (
        select(HeadTailLabel)
        .where(HeadTailLabel.label_studio_task_id == label_studio_id)
        .where(HeadTailLabel.superseded == False)
    )

    label = (await session.exec(query)).first()
    if label is None:
        logger.warning(
            "Head-tail label for Label Studio id=%d not found", label_studio_id
        )
        raise HTTPException(status_code=404, detail="Label not found")
    return label


def _gate_scan(*, judged: bool):
    """Correlated EXISTS body: does this project hold a live-labelled image
    whose laser prediction has (or has not) been judged by the gate?

    Correlated explicitly on `LaserLabel` rather than relying on
    auto-correlation, which only reaches the immediately enclosing SELECT —
    the same trap that 500'd both cohort selectors when
    `_resolved_laser_extrinsics_id()` compiled to a cross join. The inner
    label table is aliased so it cannot be mistaken for the outer one.

    `superseded == False` is applied inside the scan for the same reason
    every other laser read applies it: a dead-lettered label is not live
    labeling work, so a pending prediction on its image must not keep an
    otherwise-finished project off the landing page.
    """
    inner = alias(LaserLabel.__table__, name="gate_scan_label")
    verdict = (
        LaserPrediction.gate_verdict != None
        if judged
        else LaserPrediction.gate_verdict == None
    )
    return (
        select(1)
        .select_from(
            inner.join(
                LaserPrediction.__table__,
                LaserPrediction.image_id == inner.c.image_id,
            )
        )
        .where(inner.c.label_studio_project_id == LaserLabel.label_studio_project_id)
        .where(inner.c.superseded == False)
        .where(verdict)
        .correlate(LaserLabel.__table__)
        .exists()
    )


@app.get("/api/v1/labels/laser/label-studio-project-ids")
async def get_laser_label_studio_project_ids(
    incomplete: bool = False,
    gated: bool | None = None,
    session: AsyncSession = Depends(get_async_session),
) -> List[int]:
    """Distinct Label Studio project IDs that have at least one laser label.

    `incomplete=true` narrows to projects that have at least one label
    where `completed` is NULL or false. Backs the `apps/fishsense-lite-web/` SSR
    landing page, which surfaces only LS projects with outstanding
    labeling work.

    `gated=true` further narrows to projects the auto-accept gate has
    **finished** with: no laser prediction still awaiting a verdict, and at
    least one already judged. `gated=false` is its exact complement, and
    omitting it keeps every project.

    The predicate is "the gate is done here", not "the gate has run here",
    and the difference is the whole point of the flag. A dive swept halfway
    satisfies the weaker test while still holding frames the gate is about
    to take, so a labeler sent there does work the machine is about to
    finish — which is what this exists to prevent. A project the gate has
    never touched is likewise not gated: vacuous truth reads as False, the
    same convention `dive_pipeline_status`'s `*_labeling_complete` flags use.

    This is self-correcting across re-prediction. The persist activity
    constructs `LaserPrediction` without the gate fields and the upsert
    merges the whole model, so re-predicting a dive clears its verdicts —
    and the dive correctly drops off the landing page until the gate has
    been back through it.

    Only laser has a gate today; `LaserPrediction` is the sole prediction
    model carrying `gate_verdict`. The headtail / species / dive-slate
    endpoints deliberately take no `gated` flag rather than accepting one
    that could never be satisfied.

    Replaces a per-dive fan-out the api-workflow-worker used to do (one
    HTTP round trip per canonical dive) — that approach blew past the
    activity's 10-minute schedule_to_close timeout as the dataset grew.

    NOTE: must precede the `/api/v1/labels/laser/{image_id}` route —
    Starlette's default path converter treats `{image_id}` as `[^/]+`,
    so registration order is what disambiguates the literal segment.
    """
    logger.debug(
        "Retrieving distinct Label Studio project IDs with laser labels "
        "(incomplete=%s, gated=%s)",
        incomplete,
        gated,
    )
    query = (
        select(LaserLabel.label_studio_project_id).where(
            LaserLabel.label_studio_project_id != None
        )
        # Dead-lettered rows aren't live labeling work — mirror every other
        # laser read so a superseded-only project drops off the landing page
        # and the sync enumeration.
        .where(LaserLabel.superseded == False)
    )
    if incomplete:
        query = query.where(
            (LaserLabel.completed == False)
            | (LaserLabel.completed.is_(None))  # pylint: disable=no-member
        )
    if gated is not None:
        finished = _gate_scan(judged=True) & ~_gate_scan(judged=False)
        query = query.where(finished if gated else ~finished)
    return list((await session.exec(query.distinct())).all())


@app.get("/api/v1/labels/laser/dives-with-complete-labeling")
async def get_dives_with_complete_laser_labeling(
    session: AsyncSession = Depends(get_async_session),
) -> List[int]:
    """Dive IDs whose laser labeling is fully complete.

    A dive qualifies iff every non-superseded `LaserLabel` for one of
    its images has `completed=True` AND at least one such label exists.
    Dives with zero non-superseded laser labels (no labeling activity at
    all) are excluded — there's nothing to validate.

    Backs the laser label-validation pass that runs after each laser
    sync: validating against an in-progress dive's labels is wasted
    effort because the line fit changes as more positives arrive.

    Implemented as `(dive has at least one completed non-superseded
    laser label) AND NOT (dive has any incomplete non-superseded laser
    label)` — `NOT EXISTS` is portable across the prod Postgres and
    the in-memory sqlite the integration tests use; `bool_and` would
    work on Postgres only.

    NOTE: must precede the `/api/v1/labels/laser/{image_id}` route —
    Starlette's default path converter treats `{image_id}` as `[^/]+`,
    so registration order is what disambiguates the literal segment.
    """
    logger.debug("Retrieving dive IDs with complete laser labeling")
    incomplete = alias(LaserLabel.__table__, name="incomplete_laser_label")
    incomplete_image = alias(Image.__table__, name="incomplete_image")
    has_incomplete = (
        select(incomplete.c.id)
        .select_from(
            incomplete.join(
                incomplete_image, incomplete.c.image_id == incomplete_image.c.id
            )
        )
        .where(incomplete_image.c.dive_id == Image.dive_id)
        .where(incomplete.c.superseded == False)
        .where(
            (incomplete.c.completed == False)
            | (incomplete.c.completed.is_(None))  # pylint: disable=no-member
        )
        .exists()
    )
    query = (
        select(Image.dive_id)
        .join(LaserLabel, LaserLabel.image_id == Image.id)
        .where(LaserLabel.superseded == False)
        .where(LaserLabel.completed == True)
        .where(Image.dive_id != None)
        .where(~has_incomplete)
        .distinct()
    )
    return list((await session.exec(query)).all())


@app.get("/api/v1/labels/laser/{image_id}")
async def get_laser_label(
    image_id: int, session: AsyncSession = Depends(get_async_session)
) -> LaserLabel | None:
    """Retrieve a laser label for a given image ID."""
    logger.debug("Retrieving laser label for image with id=%d", image_id)

    query = (
        select(LaserLabel)
        .where(LaserLabel.image_id == image_id)
        .where(LaserLabel.superseded == False)
    )

    label = (await session.exec(query)).first()
    if label is None:
        logger.warning("Laser label for image with id=%d not found", image_id)
        raise HTTPException(status_code=404, detail="Label not found")
    return label


@app.get("/api/v1/labels/laser/label-studio/{label_studio_id}")
async def get_laser_label_by_label_studio_id(
    label_studio_id: int, session: AsyncSession = Depends(get_async_session)
) -> LaserLabel | None:
    """Retrieve a laser label for a given Label Studio ID."""
    logger.debug("Retrieving laser label for Label Studio id=%d", label_studio_id)

    query = (
        select(LaserLabel)
        .where(LaserLabel.label_studio_task_id == label_studio_id)
        .where(LaserLabel.superseded == False)
    )

    label = (await session.exec(query)).first()
    if label is None:
        logger.warning("Laser label for Label Studio id=%d not found", label_studio_id)
        raise HTTPException(status_code=404, detail="Label not found")
    return label


@app.get("/api/v1/dives/{dive_id}/labels/laser")
async def get_laser_labels_for_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> List[LaserLabel]:
    """Retrieve all laser labels for a given dive ID."""
    logger.debug("Retrieving laser labels for dive with id=%d", dive_id)
    query = (
        select(LaserLabel)
        .join_from(LaserLabel, Image, LaserLabel.image_id == Image.id)
        .join_from(Image, Dive, Image.dive_id == Dive.id)
        .where(Dive.id == dive_id)
        .where(LaserLabel.superseded == False)
        # Ordered because callers treat the sequence as meaningful, not just
        # the set: the populate activities append in the order they receive
        # and Label Studio assigns task order from that. See
        # `test_label_lists_are_ordered`.
        .order_by(LaserLabel.image_id)
    )

    labels = (await session.exec(query)).all()
    if not labels:
        logger.warning("Laser labels for dive with id=%d not found", dive_id)
        raise HTTPException(status_code=404, detail="Labels not found")
    return labels


@app.put("/api/v1/labels/laser/{image_id}", status_code=201)
async def put_laser_label(
    image_id: int,
    label: LaserLabel,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update a laser label for a given image ID."""
    logger.debug("Creating or updating laser label for image with id=%d", image_id)
    return await _upsert_label(session, LaserLabel, image_id, label)


@app.get("/api/v1/dives/{dive_id}/labels/species")
async def get_species_labels_for_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> List[SpeciesLabel]:
    """Retrieve all species labels for a given dive ID."""
    logger.debug("Retrieving species labels for dive with id=%d", dive_id)
    query = (
        select(SpeciesLabel)
        .join_from(SpeciesLabel, Image, SpeciesLabel.image_id == Image.id)
        .join_from(Image, Dive, Image.dive_id == Dive.id)
        .where(Dive.id == dive_id)
        .where(SpeciesLabel.superseded == False)
        .order_by(SpeciesLabel.image_id)  # see get_laser_labels_for_dive
    )

    labels = (await session.exec(query)).all()
    if not labels:
        logger.warning("Species labels for dive with id=%d not found", dive_id)
        raise HTTPException(status_code=404, detail="Labels not found")
    return labels


@app.get("/api/v1/labels/species/label-studio-project-ids")
async def get_species_label_studio_project_ids(
    incomplete: bool = False,
    session: AsyncSession = Depends(get_async_session),
) -> List[int]:
    """Distinct Label Studio project IDs that have at least one species label.

    `incomplete=true` narrows to projects that have at least one label
    where `completed` is NULL or false. Backs the `apps/fishsense-lite-web/` SSR
    landing page, which surfaces only LS projects with outstanding
    labeling work.

    NOTE: must precede the `/api/v1/labels/species/{image_id}` route —
    Starlette's default path converter treats `{image_id}` as `[^/]+`,
    so registration order is what disambiguates the literal segment.
    """
    logger.debug(
        "Retrieving distinct Label Studio project IDs with species labels "
        "(incomplete=%s)",
        incomplete,
    )
    query = (
        select(SpeciesLabel.label_studio_project_id).where(
            SpeciesLabel.label_studio_project_id != None
        )
        # Exclude dead-lettered rows — parity with every other read.
        .where(SpeciesLabel.superseded == False)
    )
    if incomplete:
        query = query.where(
            (SpeciesLabel.completed == False)
            | (SpeciesLabel.completed.is_(None))  # pylint: disable=no-member
        )
    return list((await session.exec(query.distinct())).all())


@app.get("/api/v1/labels/species/{image_id}")
async def get_species_label(
    image_id: int, session: AsyncSession = Depends(get_async_session)
) -> SpeciesLabel | None:
    """Retrieve a species label for a given image ID."""
    logger.debug("Retrieving species label for image with id=%d", image_id)
    query = (
        select(SpeciesLabel)
        .where(SpeciesLabel.image_id == image_id)
        .where(SpeciesLabel.superseded == False)
    )

    label = (await session.exec(query)).first()
    if label is None:
        logger.warning("Species label for image with id=%d not found", image_id)
        raise HTTPException(status_code=404, detail="Label not found")
    return label


@app.put("/api/v1/labels/species/{image_id}", status_code=201)
async def put_species_label(
    image_id: int,
    label: SpeciesLabel,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update a species label for a given image ID."""
    logger.debug("Creating or updating species label for image with id=%d", image_id)
    return await _upsert_label(session, SpeciesLabel, image_id, label)


@app.get("/api/v1/labels/species/label-studio/{label_studio_id}")
async def get_species_label_by_label_studio_id(
    label_studio_id: int, session: AsyncSession = Depends(get_async_session)
) -> SpeciesLabel | None:
    """Retrieve a species label for a given Label Studio task ID."""
    logger.debug("Retrieving species label for Label Studio id=%d", label_studio_id)
    query = (
        select(SpeciesLabel)
        .where(SpeciesLabel.label_studio_task_id == label_studio_id)
        .where(SpeciesLabel.superseded == False)
    )

    label = (await session.exec(query)).first()
    if label is None:
        logger.warning(
            "Species label for Label Studio id=%d not found", label_studio_id
        )
        raise HTTPException(status_code=404, detail="Label not found")
    return label


@app.get("/api/v1/labels/sync-cursor/{kind}/{label_studio_project_id}")
async def get_label_studio_sync_cursor(
    kind: str,
    label_studio_project_id: int,
    session: AsyncSession = Depends(get_async_session),
) -> LabelStudioSyncCursor | None:
    """Retrieve the sync cursor for a given (kind, project) pair.

    Returns None when no cursor exists yet — the api-workflow-worker
    treats that as "first run, sync everything."
    """
    query = (
        select(LabelStudioSyncCursor)
        .where(LabelStudioSyncCursor.kind == kind)
        .where(LabelStudioSyncCursor.label_studio_project_id == label_studio_project_id)
    )
    return (await session.exec(query)).first()


@app.put("/api/v1/labels/sync-cursor/{kind}/{label_studio_project_id}", status_code=201)
async def put_label_studio_sync_cursor(
    kind: str,
    label_studio_project_id: int,
    cursor: LabelStudioSyncCursor,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Upsert the sync cursor for a given (kind, project) pair."""
    cursor = LabelStudioSyncCursor.model_validate(jsonable_encoder(cursor))
    cursor.kind = kind
    cursor.label_studio_project_id = label_studio_project_id

    if cursor.id is None:
        existing_query = (
            select(LabelStudioSyncCursor)
            .where(LabelStudioSyncCursor.kind == kind)
            .where(
                LabelStudioSyncCursor.label_studio_project_id == label_studio_project_id
            )
        )
        existing = (await session.exec(existing_query)).first()
        if existing is not None:
            cursor.id = existing.id

    cursor = await session.merge(cursor)
    await session.flush()

    return cursor.id
