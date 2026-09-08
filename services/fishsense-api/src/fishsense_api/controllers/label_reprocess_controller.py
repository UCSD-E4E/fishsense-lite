# pylint: disable=C0121
"""Reprocess-flag endpoints: raise and lower `needs_reprocess` per label kind.

Split out of `label_controller`, which had grown past pylint's 1000-line cap.
The seam is not arbitrary: `needs_reprocess` is **cohort state, not label
content**. It says "this frame's overlay JPEG should be drawn again", which is
a statement about the preprocess pipeline rather than about what a labeler
saw -- which is also why `_upsert_label` deliberately leaves it alone, and why
these verbs live on `/dives/{dive_id}/...` while the label CRUD lives on
`/labels/...`.

Route registration is a side effect of import, so this module must stay listed
in `controllers/__init__.py`; `test_label_route_registration.py` fails if it
is dropped, or if a looser route registered earlier starts shadowing these.
"""

import logging
from typing import Annotated

from fastapi import Depends, Query
from sqlalchemy import or_
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.dive_slate_label import DiveSlateLabel
from fishsense_api.models.head_tail_label import HeadTailLabel
from fishsense_api.models.image import Image
from fishsense_api.models.laser_label import LaserLabel
from fishsense_api.models.species_label import SpeciesLabel
from fishsense_api.server import app

logger = logging.getLogger(__name__)


async def _set_needs_reprocess(
    session: AsyncSession,
    dive_id: int,
    model,
    value: bool,
    only_incomplete: bool = True,
    checksums: list[str] | None = None,
) -> int:
    """Set `needs_reprocess` on a dive's labels of one kind. Returns rows touched.

    The flag lives on the label, one per kind, rather than once on `Image`: an
    image carries a different JPEG per stage, and a change to one stage's
    overlay says nothing about the other three.

    Written once and parameterised by model rather than copied per kind. The
    four `put_*_label` handlers were near-identical for a long time and scored
    zero on `duplicate-code`, because that check is textual and
    `LaserLabel.image_id == image_id` and `SpeciesLabel.image_id == image_id`
    are different strings. A green duplicate-code run says nothing about this
    shape, so it has to be avoided by hand.

    **`only_incomplete` applies only when raising the flag.** Re-rendering a
    frame someone has already answered buys nothing, and completed rows
    outnumber open ones by more than an order of magnitude, so the default
    keeps a dive-wide flag from costing hours of NAS staging to redraw frames
    nobody will look at again.

    Clearing deliberately ignores `only_incomplete` and lowers the flag whatever
    the label's state. If it inherited that filter, a label completed *between*
    the flag being raised and the redraw finishing would keep its flag up — and
    a flag nothing lowers holds its dive in the cohort forever, re-staging raw
    `.ORF`s from the NAS every hour and starving every higher-id dive behind it
    (prod dive 60 did exactly that to dives 84/465/471 until 2026-08-04).

    **`checksums` scopes the clear to named frames**, and is how the parents
    avoid discarding a request they never acted on: their child can run for two
    hours, so a flag raised inside that window would otherwise be lowered by a
    run that redrew nothing for it. The success path passes what it redrew.

    `None` means "no scope" and clears the whole dive — the no-work backstop,
    where the flag reached no image and nothing will ever lower it. `[]` is a
    real, empty scope and clears nothing; a falsy check would collapse the two
    and turn the commonest no-work payload into a dive-wide clear.

    Canonical images only, both directions: the same physical frame lives under
    several dive rows and only the canonical copy is ever preprocessed, so
    flagging the rest would raise a flag no cohort can lower.

    Not conditioned on the current value, so both directions are idempotent.
    """
    query = (
        select(model)
        .join_from(model, Image, model.image_id == Image.id)
        .where(Image.dive_id == dive_id)
        .where(
            Image.is_canonical == True
        )  # noqa: E712  pylint: disable=singleton-comparison
    )
    if value:
        # Never raise a flag on a row the resolver cannot see.
        # `get_<kind>_labels_for_dive` -- the per-dive getter every resolver
        # reads -- filters `superseded == False`, so a flag on anything else
        # would be visible to the cohort selector and invisible to the
        # resolver: the dive is picked, its raw `.ORF`s are staged from the
        # NAS, nothing resolves, and it happens again next hour.
        #
        # `== False`, not "not superseded", and it must stay byte-identical to
        # the getter's filter. `laserlabel` and `headtaillabel` gained this
        # column nullable with no backfill (b3a78115ba3d, 06886d4ca175 --
        # unlike the species/dive-slate pair in 7934e62a12c0), so prod holds
        # NULL rows, and `NULL == False` is NULL in SQL. Reading NULL as live
        # would flag exactly the rows the resolver drops. The cost: a legacy
        # NULL row cannot be redrawn, and the endpoint returns 0 for it.
        #
        # Clearing is deliberately wider -- unfiltered -- so a row superseded
        # *after* being flagged still gets its flag lowered.
        query = query.where(
            model.superseded == False
        )  # noqa: E712  pylint: disable=singleton-comparison
        if only_incomplete:
            query = query.where(
                or_(
                    model.completed == False, model.completed.is_(None)
                )  # noqa: E712  pylint: disable=singleton-comparison
            )
    if checksums is not None:
        query = query.where(Image.checksum.in_(checksums))  # pylint: disable=no-member
    labels = (await session.exec(query)).all()
    for label in labels:
        label.needs_reprocess = value
        session.add(label)
    await session.flush()
    logger.info(
        "set needs_reprocess=%s on %d %s labels for dive_id=%d "
        "(only_incomplete=%s scoped_to=%s)",
        value,
        len(labels),
        model.__name__,
        dive_id,
        only_incomplete,
        "whole dive" if checksums is None else f"{len(checksums)} frames",
    )
    return len(labels)


@app.put("/api/v1/dives/{dive_id}/labels/laser/needs-reprocess")
async def set_laser_labels_needs_reprocess(
    dive_id: int,
    only_incomplete: bool = True,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Flag this dive's laser labels for a stage 0.1 redraw.

    Raises the flag `select_next_for_laser_preprocessing` selects on, so an
    already-preprocessed dive re-enters the cohort and its overlay JPEGs are
    regenerated at the same object-store keys. Label Studio presigns those keys
    at serve time, so existing tasks pick the new image up with no re-import
    and no loss of the labels already on them.

    `only_incomplete` defaults true — a frame someone has already answered does
    not need redrawing. See `_set_needs_reprocess`.
    """
    return await _set_needs_reprocess(
        session, dive_id, LaserLabel, True, only_incomplete=only_incomplete
    )


@app.delete("/api/v1/dives/{dive_id}/labels/laser/needs-reprocess")
async def clear_laser_labels_needs_reprocess(
    dive_id: int,
    checksums: Annotated[list[str] | None, Query()] = None,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Lower the flag once this dive's laser JPEGs have been redrawn.

    Called by the stage 0.1 parent after its data-worker child completes. This is
    the half that keeps the cohort drainable.

    `checksums` scopes the clear to the frames actually redrawn; omitting it
    clears the whole dive. See `_set_needs_reprocess`. A dive with no laser labels
    returns 0 rather than 404: the parent calls it unconditionally.
    """
    return await _set_needs_reprocess(
        session, dive_id, LaserLabel, False, checksums=checksums
    )


@app.put("/api/v1/dives/{dive_id}/labels/species/needs-reprocess")
async def set_species_labels_needs_reprocess(
    dive_id: int,
    only_incomplete: bool = True,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Flag this dive's species labels for a stage 2 redraw.

    Raises the flag `select_next_for_species_preprocessing` selects on, so an
    already-preprocessed dive re-enters the cohort and its overlay JPEGs are
    regenerated at the same object-store keys. Label Studio presigns those keys
    at serve time, so existing tasks pick the new image up with no re-import
    and no loss of the labels already on them.

    `only_incomplete` defaults true — a frame someone has already answered does
    not need redrawing. See `_set_needs_reprocess`.
    """
    return await _set_needs_reprocess(
        session, dive_id, SpeciesLabel, True, only_incomplete=only_incomplete
    )


@app.delete("/api/v1/dives/{dive_id}/labels/species/needs-reprocess")
async def clear_species_labels_needs_reprocess(
    dive_id: int,
    checksums: Annotated[list[str] | None, Query()] = None,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Lower the flag once this dive's species JPEGs have been redrawn.

    Called by the stage 2 parent after its data-worker child completes. This is
    the half that keeps the cohort drainable.

    `checksums` scopes the clear to the frames actually redrawn; omitting it
    clears the whole dive. See `_set_needs_reprocess`. A dive with no species labels
    returns 0 rather than 404: the parent calls it unconditionally.
    """
    return await _set_needs_reprocess(
        session, dive_id, SpeciesLabel, False, checksums=checksums
    )


@app.put("/api/v1/dives/{dive_id}/labels/headtail/needs-reprocess")
async def set_headtail_labels_needs_reprocess(
    dive_id: int,
    only_incomplete: bool = True,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Flag this dive's headtail labels for a stage 5.1 redraw.

    Raises the flag `select_next_for_headtail_preprocessing` selects on, so an
    already-preprocessed dive re-enters the cohort and its overlay JPEGs are
    regenerated at the same object-store keys. Label Studio presigns those keys
    at serve time, so existing tasks pick the new image up with no re-import
    and no loss of the labels already on them.

    `only_incomplete` defaults true — a frame someone has already answered does
    not need redrawing. See `_set_needs_reprocess`.
    """
    return await _set_needs_reprocess(
        session, dive_id, HeadTailLabel, True, only_incomplete=only_incomplete
    )


@app.delete("/api/v1/dives/{dive_id}/labels/headtail/needs-reprocess")
async def clear_headtail_labels_needs_reprocess(
    dive_id: int,
    checksums: Annotated[list[str] | None, Query()] = None,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Lower the flag once this dive's headtail JPEGs have been redrawn.

    Called by the stage 5.1 parent after its data-worker child completes. This is
    the half that keeps the cohort drainable.

    `checksums` scopes the clear to the frames actually redrawn; omitting it
    clears the whole dive. See `_set_needs_reprocess`. A dive with no headtail labels
    returns 0 rather than 404: the parent calls it unconditionally.
    """
    return await _set_needs_reprocess(
        session, dive_id, HeadTailLabel, False, checksums=checksums
    )


@app.put("/api/v1/dives/{dive_id}/labels/dive-slate/needs-reprocess")
async def set_dive_slate_labels_needs_reprocess(
    dive_id: int,
    only_incomplete: bool = True,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Flag this dive's dive-slate labels for a stage 9 redraw.

    Raises the flag `select_next_for_dive_slate_preprocessing` selects on, so an
    already-preprocessed dive re-enters the cohort and its overlay JPEGs are
    regenerated at the same object-store keys. Label Studio presigns those keys
    at serve time, so existing tasks pick the new image up with no re-import
    and no loss of the labels already on them.

    `only_incomplete` defaults true — a frame someone has already answered does
    not need redrawing. See `_set_needs_reprocess`.
    """
    return await _set_needs_reprocess(
        session, dive_id, DiveSlateLabel, True, only_incomplete=only_incomplete
    )


@app.delete("/api/v1/dives/{dive_id}/labels/dive-slate/needs-reprocess")
async def clear_dive_slate_labels_needs_reprocess(
    dive_id: int,
    checksums: Annotated[list[str] | None, Query()] = None,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Lower the flag once this dive's dive-slate JPEGs have been redrawn.

    Called by the stage 9 parent after its data-worker child completes. This is
    the half that keeps the cohort drainable.

    `checksums` scopes the clear to the frames actually redrawn; omitting it
    clears the whole dive. See `_set_needs_reprocess`. A dive with no dive-slate labels
    returns 0 rather than 404: the parent calls it unconditionally.
    """
    return await _set_needs_reprocess(
        session, dive_id, DiveSlateLabel, False, checksums=checksums
    )
