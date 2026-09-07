"""Species / head/tail / slate cohorts honour `needs_reprocess`, like laser does.

The flag has been on all four label tables since `c7e4a91f2d38`, but only
stage 0.1 ever read it. For the other three, a raised flag did nothing at all:
the cohort predicate is "image has no row of this kind", which goes false the
moment populate seeds a row, so an already-preprocessed image was unreachable
and its JPEG frozen.

**Selector and resolver must move together.** A selector that honours the flag
while its resolver ignores it picks the dive, stages the dive's raw `.ORF`s
from the NAS, resolves zero images, and does the same thing again next hour --
forever. CLAUDE.md requires resolvers to mirror selector predicates exactly,
and this is the failure that rule exists to prevent. The resolver side is
covered in the api-worker's own tests.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession


@pytest.fixture
async def session():
    import fishsense_api.database  # noqa: F401  pylint: disable=unused-import

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


_WHEN = datetime(2026, 9, 1, tzinfo=timezone.utc)


async def _seed_labelled_dive(session, dive_id, kind, *, flagged, canonical=True):
    """A dive whose every canonical image already carries a real label row.

    Without a flag this dive is out of every preprocess cohort by construction:
    the "no row of this kind" term is false for all its images.
    """
    from fishsense_api.models.dive import Dive
    from fishsense_api.models.dive_slate_label import DiveSlateLabel
    from fishsense_api.models.head_tail_label import HeadTailLabel
    from fishsense_api.models.image import Image
    from fishsense_api.models.laser_label import LaserLabel
    from fishsense_api.models.priority import Priority
    from fishsense_api.models.species_label import SpeciesLabel

    session.add(
        Dive(
            id=dive_id,
            path=f"/d{dive_id}",
            priority=Priority.HIGH,
            dive_datetime=_WHEN,
            dive_slate_id=1,
        )
    )
    image_id = dive_id * 100
    session.add(
        Image(
            id=image_id,
            dive_id=dive_id,
            path=f"/i{image_id}.ORF",
            checksum=f"{image_id:032d}",
            is_canonical=canonical,
            taken_datetime=_WHEN,
        )
    )
    # A valid laser label: the gate species and head/tail both cascade from.
    session.add(
        LaserLabel(
            image_id=image_id,
            label_studio_project_id=1,
            completed=True,
            superseded=False,
            x=10.0,
            y=20.0,
        )
    )
    # Slate frames are identified by their species content_of_image marker.
    session.add(
        SpeciesLabel(
            image_id=image_id,
            label_studio_project_id=1,
            completed=True,
            content_of_image="Slate, Laser on slate",
            needs_reprocess=flagged if kind == "species" else False,
        )
    )
    if kind == "headtail":
        session.add(
            HeadTailLabel(
                image_id=image_id,
                label_studio_project_id=1,
                completed=True,
                superseded=False,
                needs_reprocess=flagged,
            )
        )
    else:
        session.add(
            HeadTailLabel(
                image_id=image_id,
                label_studio_project_id=1,
                completed=True,
                superseded=False,
            )
        )
    if kind == "dive-slate":
        session.add(
            DiveSlateLabel(
                image_id=image_id,
                label_studio_project_id=1,
                completed=True,
                needs_reprocess=flagged,
            )
        )
    else:
        session.add(
            DiveSlateLabel(
                image_id=image_id, label_studio_project_id=1, completed=True
            )
        )
    await session.commit()


def _models():
    from fishsense_api.models.dive_slate_label import DiveSlateLabel
    from fishsense_api.models.head_tail_label import HeadTailLabel
    from fishsense_api.models.species_label import SpeciesLabel

    return {
        "species": SpeciesLabel,
        "headtail": HeadTailLabel,
        "dive-slate": DiveSlateLabel,
    }


#: Physical table per kind, for the raw-SQL NULL seed below.
_TABLES = {
    "species": "specieslabel",
    "headtail": "headtaillabel",
    "dive-slate": "diveslatelabel",
}

_SELECTORS = {
    "species": "select_next_for_species_preprocessing",
    "headtail": "select_next_for_headtail_preprocessing",
    "dive-slate": "select_next_for_slate_preprocessing",
}


@pytest.mark.parametrize("kind", sorted(_SELECTORS))
class TestCohortHonoursFlag:
    async def test_fully_labelled_dive_is_not_selected_without_a_flag(
        self, session, kind
    ):
        """The control. If this ever fails the test below proves nothing."""
        from fishsense_api.controllers import dive_cohort_controller

        await _seed_labelled_dive(session, 1, kind, flagged=False)
        selector = getattr(dive_cohort_controller, _SELECTORS[kind])
        assert await selector(session=session) is None

    async def test_flagged_dive_is_selected(self, session, kind):
        from fishsense_api.controllers import dive_cohort_controller

        await _seed_labelled_dive(session, 1, kind, flagged=True)
        selector = getattr(dive_cohort_controller, _SELECTORS[kind])
        assert await selector(session=session) == 1

    async def test_flag_on_a_non_canonical_image_does_not_select(self, session, kind):
        """Only the canonical copy is ever preprocessed, so a flag on a
        duplicate would select a dive the resolver finds no work for -- and the
        dive would re-stage its raw bytes from the NAS every hour forever."""
        from fishsense_api.controllers import dive_cohort_controller

        await _seed_labelled_dive(session, 1, kind, flagged=True, canonical=False)
        selector = getattr(dive_cohort_controller, _SELECTORS[kind])
        assert await selector(session=session) is None

    async def test_flag_on_a_null_superseded_row_does_not_select(self, session, kind):
        """The other half of the raise-path guard.

        Every resolver reads its labels through `get_<kind>_labels_for_dive`,
        which filters `superseded == False` -- and NULL is not False in SQL. So
        a NULL row is invisible to the resolver, and a selector that counted it
        as live would pick this dive, stage its raw `.ORF`s from the NAS,
        resolve nothing, and do it again every hour forever.

        `laserlabel` and `headtaillabel` still hold NULLs in prod: both gained
        the column nullable with no backfill (b3a78115ba3d, 06886d4ca175),
        unlike the species/dive-slate pair (7934e62a12c0). The NULL is written
        in SQL because `Field(default=False)` is a column default, so the ORM
        substitutes False on insert and cannot produce the legacy row at all.
        """
        from sqlmodel import text

        from fishsense_api.controllers import dive_cohort_controller

        # The dive keeps its live, unflagged row, so the primary "image has no
        # row of this kind" term stays false and only the flag could select it.
        # A second, legacy row on the same image carries the flag and the NULL.
        await _seed_labelled_dive(session, 1, kind, flagged=False)
        # A different project: `(image_id, label_studio_project_id)` is the
        # natural key, and a legacy row would in practice sit in one of the
        # grandfathered shared projects anyway.
        legacy = _models()[kind](
            image_id=100, label_studio_project_id=2, completed=False
        )
        session.add(legacy)
        await session.commit()
        await session.exec(
            text(
                f"UPDATE {_TABLES[kind]} "
                "SET superseded = NULL, needs_reprocess = 1 WHERE id = :i"
            ).bindparams(i=legacy.id)
        )

        selector = getattr(dive_cohort_controller, _SELECTORS[kind])
        assert await selector(session=session) is None
