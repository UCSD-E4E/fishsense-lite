"""Only *canonical* images are pipeline work.

The same physical frames legitimately live under several dive rows — prod has
131,430 image rows over 65,981 distinct checksums, so **half the table is
duplicate content**, and 207 of 479 dives are duplicates end to end.
`is_canonical` marks which copy is the real one (first row for a checksum wins,
from `9e5bc64`).

Today nothing in the pipeline reads that flag: `is_canonical` appears in
exactly one query in the whole codebase (`get_canonical_dives`). Duplicate
dives are inert only because they all happen to be `priority=LOW`, and every
cohort gates on HIGH. That is a coincidence of the data, not a property of the
code — promote one duplicate dive and it starts consuming NAS bandwidth,
preprocessing, and labeler time on frames that already exist elsewhere.

**Gating populate alone would make it worse, not better.** The preprocess
cohorts are "at least one image with no label row". If populate skipped
non-canonical images, those rows would never get labels, so the dive would
never drain — re-staging its raw `.ORF`s from the NAS every hour and blocking
every higher-id dive behind it. That is the prod dive 60 wedge, verbatim. So
the cohorts have to exclude them too, which is what these tests pin.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession


@pytest.fixture
async def session():
    import fishsense_api.database  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _dive(dive_id: int, **kwargs):
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    kwargs.setdefault("priority", Priority.HIGH)
    return Dive(
        id=dive_id,
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        **kwargs,
    )


def _image(image_id: int, dive_id: int, *, is_canonical: bool):
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    return Image(
        id=image_id,
        path=f"/dev/null/img-{image_id}",
        taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        checksum=f"{image_id:032d}",
        is_canonical=is_canonical,
        dive_id=dive_id,
    )


# ── stage 0.1: the wedge this prevents ────────────────────────────────


async def test_a_dive_of_only_duplicate_frames_is_not_laser_preprocessing_work(session):
    """The dive-60 wedge, in its duplicate-frame form.

    Cohort is "HIGH + an image with no LaserLabel row". A promoted duplicate
    dive satisfies it forever: populate would (correctly) decline to make LS
    tasks for frames that already exist elsewhere, so no label row ever
    appears, so the dive never drains — re-staging raw `.ORF`s from the NAS
    every hour and blocking every higher-id dive.
    """
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )

    session.add(_dive(1))
    session.add(_image(11, 1, is_canonical=False))
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) is None


async def test_a_canonical_frame_is_still_laser_preprocessing_work(session):
    """The other half — the gate must not swallow real work."""
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )

    session.add(_dive(1))
    session.add(_image(11, 1, is_canonical=True))
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) == 1


async def test_a_mixed_dive_is_still_work_for_its_canonical_frames(session):
    """A dive holding one original and one duplicate is real work. Excluding
    the whole dive would strand the original."""
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )

    session.add(_dive(1))
    session.add(_image(11, 1, is_canonical=False))
    session.add(_image(12, 1, is_canonical=True))
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) == 1


# ── the registry property: every selector, not just the one ───────────


def test_every_cohort_selector_filters_on_is_canonical():
    """A source-level guard, in the spirit of `controllers/__init__.py` and the
    view registry: the failure mode is silent.

    A new selector that subqueries `Image` without this filter reintroduces the
    wedge, and nothing at runtime would say so — the dive simply never drains.
    Every `Image.dive_id == Dive.id` correlation must be paired with an
    `Image.is_canonical` predicate.
    """
    import inspect  # pylint: disable=import-outside-toplevel
    import re  # pylint: disable=import-outside-toplevel

    from fishsense_api.controllers import (  # pylint: disable=import-outside-toplevel
        dive_cohort_controller,
        dive_controller,
    )

    # Both modules, deliberately: the selectors live in `dive_cohort_controller`
    # but `dive_controller` still correlates Image to Dive (canonical dives), and
    # a future selector could land in either. Checking only where they happen to
    # live today is how this guard would quietly stop guarding.
    source = "\n".join(
        inspect.getsource(module)
        for module in (dive_cohort_controller, dive_controller)
    )
    correlations = len(re.findall(r"Image\.dive_id == Dive\.id", source))
    gated = len(re.findall(r"Image\.is_canonical == True", source))

    assert correlations > 0, "sanity: selectors should correlate Image to Dive"
    assert gated >= correlations, (
        f"{correlations} `Image.dive_id == Dive.id` correlations but only "
        f"{gated} `Image.is_canonical` filters — a selector is missing the "
        "gate and would wedge on a duplicate dive"
    )


# ── the view must agree with the selectors ────────────────────────────


async def _pipeline_row(session, dive_id: int):
    from sqlalchemy import text  # pylint: disable=import-outside-toplevel

    from fishsense_api.views import (  # pylint: disable=import-outside-toplevel
        DIVE_PIPELINE_STATUS_VIEW_SQL,
    )

    await session.exec(text(DIVE_PIPELINE_STATUS_VIEW_SQL))
    result = await session.exec(
        text(
            "SELECT laser_preprocessed, headtail_preprocessed "
            f"FROM dive_pipeline_status WHERE dive_id = {dive_id}"
        )
    )
    row = result.first()
    # SQLite hands back 1/0 for view booleans, not Python bools — the same
    # coercion `test_dive_pipeline_status_view.py` does.
    return None if row is None else tuple(bool(v) for v in row)


async def test_the_view_ignores_duplicate_frames_like_the_selectors_do(session):
    """The view and the cohort selectors are the same predicate in two files,
    and CLAUDE.md requires them to stay in step.

    Without this, a duplicate dive reads as perpetually incomplete on the
    Superset dashboard while the worker correctly never picks it up — the
    silent disagreement the cross-controller drift guard exists to catch.

    A dive of only duplicate frames must look the way a dive with no images
    looks: vacuously False, nothing outstanding.
    """
    session.add(_dive(1))
    session.add(_image(11, 1, is_canonical=False))
    await session.flush()

    row = await _pipeline_row(session, 1)

    # `laser_preprocessed` requires "an image exists AND none lack a label".
    # With the duplicate excluded there is no image, so it is vacuously False —
    # exactly as for an empty dive, and consistent with the selector returning
    # None rather than treating it as outstanding work.
    assert row[0] is False


async def test_the_view_still_reports_canonical_frames(session):
    """The gate must not blind the dashboard to real work."""
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    session.add(_image(11, 1, is_canonical=True))
    await session.flush()
    session.add(LaserLabel(image_id=11, label_studio_project_id=73))
    await session.flush()

    row = await _pipeline_row(session, 1)

    assert row[0] is True   # every canonical image has a laser label row


async def test_view_and_selector_agree_on_a_duplicate_dive(session):
    """The property, stated directly: for the same dive, the view must not
    claim outstanding work that the selector refuses to schedule."""
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )

    session.add(_dive(1))
    session.add(_image(11, 1, is_canonical=False))
    await session.flush()

    selected = await select_next_for_laser_preprocessing(session=session)
    row = await _pipeline_row(session, 1)

    assert selected is None          # worker: nothing to do
    assert row[0] is False           # view: nothing claimed as done either
