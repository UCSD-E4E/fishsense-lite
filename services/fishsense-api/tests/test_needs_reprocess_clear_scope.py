"""Clearing must not discard a flag the run never acted on.

The parent clears after its data-worker child completes, and that child can run
for up to two hours. An operator who raises a flag inside that window has it
silently lowered at the end of the run, having redrawn nothing -- the request
is gone, with no error and no retry, and the only sign is that the frame still
looks the way it did.

So the success path clears exactly the frames it redrew, named by checksum.
Anything flagged since stays flagged and is picked up on the next firing.

The no-work path still clears everything on purpose, and that asymmetry is the
point: there, the flag reached no image at all, so there is nothing to redraw
and nothing that will ever lower it. Leaving it up holds the dive in the cohort
forever, re-staging its raw `.ORF`s from the NAS every hour and starving every
higher-id dive behind it. Losing a request is recoverable; a wedge is not.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from tests_support.db import reprocess_label_kinds


_REDRAWN = f"{10:032d}"
_RAISED_MID_RUN = f"{11:032d}"


async def _seed_two_flagged_images(session, model):
    """Two flagged frames: one the run redrew, one flagged while it ran."""
    from fishsense_api.models.dive import Dive
    from fishsense_api.models.image import Image
    from fishsense_api.models.priority import Priority

    when = datetime(2026, 9, 1, tzinfo=timezone.utc)
    session.add(Dive(id=1, path="/d1", priority=Priority.HIGH, dive_datetime=when))
    for image_id in (10, 11):
        session.add(
            Image(
                id=image_id,
                dive_id=1,
                path=f"/i{image_id}.ORF",
                checksum=f"{image_id:032d}",
                is_canonical=True,
                taken_datetime=when,
            )
        )
        session.add(
            model(
                image_id=image_id,
                completed=False,
                label_studio_project_id=5,
                needs_reprocess=True,
            )
        )
    await session.flush()


async def _flags(session, model) -> set[str]:
    """Checksums whose flag is still raised.

    Filtered in Python rather than SQL: `needs_reprocess == True` in a WHERE
    clause trips `singleton-comparison`, and the usual inline pragma is
    fragile here because black reflows the statement and moves the comment off
    the line it applies to.
    """
    from sqlmodel import select

    from fishsense_api.models.image import Image

    rows = (
        await session.exec(
            select(Image.checksum, model.needs_reprocess).join(
                model, model.image_id == Image.id
            )
        )
    ).all()
    return {checksum for checksum, flag in rows if flag}


@pytest.mark.parametrize("model", reprocess_label_kinds())
class TestClearScope:
    async def test_scoped_clear_leaves_a_flag_raised_during_the_run(
        self, session, model
    ):
        from fishsense_api.controllers.label_controller import _set_needs_reprocess

        await _seed_two_flagged_images(session, model)
        n = await _set_needs_reprocess(
            session, 1, model, False, checksums=[_REDRAWN]
        )

        assert n == 1
        assert await _flags(session, model) == {_RAISED_MID_RUN}

    async def test_unscoped_clear_still_lowers_everything(self, session, model):
        """The no-work backstop. Without this the dive can never drain."""
        from fishsense_api.controllers.label_controller import _set_needs_reprocess

        await _seed_two_flagged_images(session, model)
        n = await _set_needs_reprocess(session, 1, model, False)

        assert n == 2
        assert await _flags(session, model) == set()

    async def test_an_empty_scope_is_not_read_as_no_scope(self, session, model):
        """`[]` means "this run redrew nothing", which must clear nothing.

        Falsy-checking the list instead would turn the most common no-work
        payload into a dive-wide clear -- the exact thing the scope exists to
        prevent, reached by accident.
        """
        from fishsense_api.controllers.label_controller import _set_needs_reprocess

        await _seed_two_flagged_images(session, model)
        n = await _set_needs_reprocess(session, 1, model, False, checksums=[])

        assert n == 0
        assert await _flags(session, model) == {_REDRAWN, _RAISED_MID_RUN}
