"""Flagging a dive for re-render must not re-render what is already labelled.

`needs_reprocess` puts an already-preprocessed image back into its stage's
cohort so the overlay JPEG is redrawn at the same object-store key — Label
Studio presigns that key at serve time, so an existing task shows the new
pixels with no re-import and no loss of work already done.

The reason the flag needs a scope: re-rendering a frame a labeler has already
answered buys nothing, and the completed rows outnumber the open ones by more
than an order of magnitude (prod 2026-09: 46,417 completed laser and 36,607
completed head/tail rows against 259 and 3,147 open). Flagging a dive
wholesale therefore costs hours of NAS staging and rectification to redraw
frames nobody will look at again.

So the default is incomplete-only, and `only_incomplete=False` is the explicit
opt-in for the rare case where the completed frames genuinely need redrawing
too (a rendering bug that invalidates the labels themselves, say).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from tests_support.db import reprocess_label_kinds


async def _seed(session, model):
    """One dive: an open label, a completed label, and a non-canonical open one."""
    from fishsense_api.models.dive import Dive
    from fishsense_api.models.image import Image
    from fishsense_api.models.priority import Priority

    when = datetime(2026, 9, 1, tzinfo=timezone.utc)
    session.add(Dive(id=1, path="/d1", priority=Priority.HIGH, dive_datetime=when))
    rows = {}
    for image_id, completed, canonical in (
        (10, False, True),   # open, canonical      -> flagged
        (11, True, True),    # completed, canonical -> NOT flagged by default
        (12, False, False),  # open, non-canonical  -> never flagged
    ):
        session.add(
            Image(
                id=image_id,
                dive_id=1,
                path=f"/i{image_id}.ORF",
                checksum=f"{image_id:032d}",
                is_canonical=canonical,
                taken_datetime=when,
            )
        )
        label = model(image_id=image_id, completed=completed, label_studio_project_id=5)
        session.add(label)
        rows[image_id] = label
    await session.flush()
    return rows


@pytest.mark.parametrize("model", reprocess_label_kinds())
class TestScoping:
    async def test_default_flags_only_incomplete_canonical_labels(self, session, model):
        from fishsense_api.controllers.label_reprocess_controller import _set_needs_reprocess

        rows = await _seed(session, model)
        n = await _set_needs_reprocess(session, 1, model, True)

        assert n == 1, "only the open canonical label should be flagged"
        assert rows[10].needs_reprocess is True
        assert rows[11].needs_reprocess is False, "already answered — nothing to redraw"
        assert rows[12].needs_reprocess is False, "non-canonical is never preprocessed"

    async def test_only_incomplete_false_flags_completed_too(self, session, model):
        from fishsense_api.controllers.label_reprocess_controller import _set_needs_reprocess

        rows = await _seed(session, model)
        n = await _set_needs_reprocess(session, 1, model, True, only_incomplete=False)

        assert n == 2, "both canonical labels, still never the non-canonical one"
        assert rows[10].needs_reprocess is True
        assert rows[11].needs_reprocess is True
        assert rows[12].needs_reprocess is False

    async def test_clearing_lowers_every_canonical_flag_regardless_of_completion(
        self, session, model
    ):
        """The parent clears unconditionally after its child completes. If clear
        inherited the incomplete-only scope, a label completed *between* the
        flag being raised and the redraw finishing would keep its flag up and
        hold the dive in the cohort forever."""
        from fishsense_api.controllers.label_reprocess_controller import _set_needs_reprocess

        rows = await _seed(session, model)
        await _set_needs_reprocess(session, 1, model, True, only_incomplete=False)
        n = await _set_needs_reprocess(session, 1, model, False)

        assert n == 2
        assert rows[10].needs_reprocess is False
        assert rows[11].needs_reprocess is False

    async def test_is_idempotent_in_both_directions(self, session, model):
        from fishsense_api.controllers.label_reprocess_controller import _set_needs_reprocess

        rows = await _seed(session, model)
        assert await _set_needs_reprocess(session, 1, model, True) == 1
        assert await _set_needs_reprocess(session, 1, model, True) == 1
        assert rows[10].needs_reprocess is True
        assert await _set_needs_reprocess(session, 1, model, False) == 2
        assert await _set_needs_reprocess(session, 1, model, False) == 2
        assert rows[10].needs_reprocess is False

    async def test_dive_with_no_labels_returns_zero_not_404(self, session, model):
        """The parent calls clear unconditionally; a 404 would fail the workflow."""
        from fishsense_api.controllers.label_reprocess_controller import _set_needs_reprocess

        assert await _set_needs_reprocess(session, 999, model, False) == 0


@pytest.mark.parametrize("model", reprocess_label_kinds())
class TestSupersededRowsAreNeverFlagged:
    """A superseded row is dead-lettered: no labeler will ever see it again.

    It also cannot be *reached*. `get_{kind}_labels_for_dive` -- the per-dive
    getter every resolver uses -- filters `superseded == False`, so a flag on a
    superseded row is visible to the cohort selector and invisible to the
    resolver. That is the exact selector/resolver mismatch that stages a dive's
    raw `.ORF`s from the NAS every hour and resolves nothing.
    """

    async def test_superseded_incomplete_row_is_not_flagged(self, session, model):
        from fishsense_api.controllers.label_reprocess_controller import _set_needs_reprocess
        from fishsense_api.models.dive import Dive
        from fishsense_api.models.image import Image
        from fishsense_api.models.priority import Priority

        when = datetime(2026, 9, 1, tzinfo=timezone.utc)
        session.add(Dive(id=2, path="/d2", priority=Priority.HIGH, dive_datetime=when))
        session.add(
            Image(
                id=20,
                dive_id=2,
                path="/i20.ORF",
                checksum=f"{20:032d}",
                is_canonical=True,
                taken_datetime=when,
            )
        )
        row = model(
            image_id=20,
            completed=False,
            superseded=True,
            label_studio_project_id=5,
        )
        session.add(row)
        await session.flush()

        assert await _set_needs_reprocess(session, 2, model, True) == 0
        assert row.needs_reprocess is False
