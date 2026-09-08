"""A legacy `superseded IS NULL` row must not be flagged, and must not select.

`laserlabel.superseded` (b3a78115ba3d) and `headtaillabel.superseded`
(06886d4ca175) were both added nullable with **no backfill**, unlike the
species/dive-slate pair (7934e62a12c0, which runs
`UPDATE ... SET superseded = FALSE WHERE superseded IS NULL`). So rows written
before those migrations still carry NULL in prod.

That matters because the two sides read NULL differently. Every resolver
reaches its labels through `get_<kind>_labels_for_dive`, which filters
`superseded == False` — and in SQL `NULL == False` is NULL, i.e. not matched.
So a NULL row is invisible to the resolver. If the raise path or the cohort
selector counted it as live instead, the flag would be visible to the selector
and invisible to the resolver: the dive is picked, its raw `.ORF`s are staged
from the NAS, nothing resolves, and it happens again every hour forever —
the dive-60 wedge shape from CLAUDE.md.

The two sides are therefore pinned to the same reading here. This is
deliberately the *narrow* one: it costs the ability to redraw legacy NULL rows
(the endpoint honestly returns 0 for them) and buys a guarantee that no flag
can be raised that nothing can lower. Widening it instead — backfilling the
NULLs to FALSE — would also make those rows newly visible to
`get_laser_labels_for_dive`, and so to stages 13 and 14, changing calibration
and measurement inputs in prod. That is a separate decision with its own blast
radius, not a bug fix.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from tests_support.db import reprocess_label_kinds
from sqlmodel import text


async def _seed_null_superseded(session, model):
    """One dive, one canonical open label whose `superseded` is NULL."""
    from fishsense_api.models.dive import Dive
    from fishsense_api.models.image import Image
    from fishsense_api.models.priority import Priority

    when = datetime(2026, 9, 1, tzinfo=timezone.utc)
    session.add(Dive(id=1, path="/d1", priority=Priority.HIGH, dive_datetime=when))
    session.add(
        Image(
            id=10,
            dive_id=1,
            path="/i10.ORF",
            checksum=f"{10:032d}",
            is_canonical=True,
            taken_datetime=when,
        )
    )
    label = model(image_id=10, completed=False, label_studio_project_id=5)
    session.add(label)
    await session.flush()

    # NULL has to be written in SQL, not through the model. `superseded` is
    # declared `Field(default=False)`, which is a *column* default, so
    # SQLAlchemy substitutes False on insert and the ORM cannot produce the
    # legacy row at all. The real NULLs got there the only way they can: the
    # migration added the column to rows that already existed.
    # The seed holds exactly one label, so this needs no WHERE.
    await session.exec(text(f"UPDATE {model.__tablename__} SET superseded = NULL"))
    return label


async def _flag_in_db(session, model) -> int:
    """Read `needs_reprocess` straight out of SQL.

    The seeded ORM object is deliberately stale -- the NULL was written behind
    its back -- and refreshing it would be a lazy sync load inside an async
    session.
    """
    rows = (
        await session.exec(
            text(f"SELECT needs_reprocess FROM {model.__tablename__}")
        )
    ).all()
    return sum(1 for (flag,) in rows if flag)


@pytest.mark.parametrize("model", reprocess_label_kinds())
class TestNullSuperseded:
    async def test_raising_skips_a_null_superseded_row(self, session, model):
        from fishsense_api.controllers.label_reprocess_controller import _set_needs_reprocess

        await _seed_null_superseded(session, model)
        n = await _set_needs_reprocess(session, 1, model, True)

        assert n == 0, "the resolver cannot see this row, so nothing may flag it"
        assert await _flag_in_db(session, model) == 0

    async def test_clearing_still_reaches_a_null_superseded_row(self, session, model):
        """Clearing must stay wider than raising.

        A row superseded *after* being flagged has to be able to give its flag
        back, or it holds its dive in the cohort forever.
        """
        from fishsense_api.controllers.label_reprocess_controller import _set_needs_reprocess

        label = await _seed_null_superseded(session, model)
        label.needs_reprocess = True
        session.add(label)
        await session.flush()
        # Flushing the ORM object rewrites `superseded` from its in-memory
        # value, so put the NULL back before exercising the clear.
        await session.exec(text(f"UPDATE {model.__tablename__} SET superseded = NULL"))

        n = await _set_needs_reprocess(session, 1, model, False)

        assert n == 1, "a row superseded after being flagged must still clear"
        assert await _flag_in_db(session, model) == 0
