"""Known-size reference for the physical fish models + the accuracy view.

The models (and the ruler) are the pipeline's held-out VALIDATION set: their
true lengths are never fed into calibration (that would be circular), only
compared against what stage 14 measures. Persisting them here turns that
comparison into a queryable artifact instead of a number in someone's notes,
and backs a Superset accuracy dashboard.

`fish_model_measurement_accuracy` joins measurements to the reference through
`Fish.name` — the same natural key stage 14 resolves model identity by — so a
new measurement shows up in the view with no extra wiring.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_shared import taxonomy

from fishsense_api.views import (
    FISH_MODEL_ACCURACY_VIEW_SQL,
    FISH_MODEL_NOTES,
    KNOWN_FISH_MODELS,
)


@pytest.fixture
async def session():
    import fishsense_api.database  # noqa: F401  pylint: disable=unused-import

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(text(FISH_MODEL_ACCURACY_VIEW_SQL))
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


# ── the seed data itself ──────────────────────────────────────────────


def test_known_models_cover_the_labeled_taxonomy():
    """EVERY `Fish Model, <name>` leaf a labeler can pick must have a known
    length, or its measurements can never be graded.

    Asserted against the whole labeled set rather than a hand-picked subset,
    because the failure this guards is *silent*: the accuracy view inner-joins
    on `Fish.name`, so a model with no reference row simply never appears. No
    error, no NULL row, no count that looks wrong — the measurements just are
    not there. `Weasly Fish` sat in the species XML in exactly that state, and
    the previous version of this test passed the whole time because it only
    checked four names it already knew about.
    """
    labeled = set(taxonomy.LABELED_FISH_MODELS)
    known = {m["name"] for m in KNOWN_FISH_MODELS}

    assert labeled <= known, f"labelable but ungradeable: {sorted(labeled - known)}"


def test_weasly_fish_records_which_landmark_its_length_uses():
    """A reference that does not say which end it means is ambiguous by ~5pp —
    larger than most of the errors being chased — so the landmark has to be on
    the record even though every reference here shares one."""
    assert "fork length" in FISH_MODEL_NOTES["Weasly Fish"].lower()


def test_the_fork_convention_is_recorded_for_the_whole_table():
    """All reference lengths are fork lengths, and that is a property of the
    SET, not of one row.

    Recorded centrally because it retires a standing hypothesis: the residual
    ladder (Purple Angel -2.2%, Shark -2.6%, Grouper -5.1%, Snook -9.6%) was
    read as a total-length-vs-fork mismatch ordered by fork depth — "one
    definition fix, not four". It cannot be, if the references were fork
    lengths all along; and a labeler clicking the tail tip instead would read
    LONG, not short. Someone will re-derive that hypothesis from the shape of
    the numbers unless the refutation sits next to them.
    """
    import inspect

    from fishsense_api import views

    src = inspect.getsource(views)
    head = src[: src.index("KNOWN_FISH_MODELS = [")]

    assert "FORK length" in head
    assert "OPEN" in head


def test_weasly_fish_widths_are_machine_reparsable():
    """The widths are the input to a separate experiment, so they need to come
    back out as numbers rather than being readable prose only."""
    import re

    notes = FISH_MODEL_NOTES["Weasly Fish"]
    found = dict(re.findall(r"(width_\w+_mm)=([0-9.]+)", notes))

    assert float(found["width_midbody_mm"]) == pytest.approx(58.69)
    assert float(found["width_caudal_peduncle_mm"]) == pytest.approx(29.56)


def test_weasly_fish_records_the_calipered_widths():
    """The round-model thickness work needs thickness as an INDEPENDENT input.

    Stage 14 measures a length from two clicked landmarks; a model's girth is
    invisible to it. If thickness were ever back-solved from measurement error
    it would absorb whatever calibration bias happened to be present, and the
    validation set would quietly start grading itself — the same circularity
    that keeps these lengths out of calibration in the first place.

    So the widths are calipered off the physical model and recorded as
    provenance, in millimetres as measured.
    """
    notes = FISH_MODEL_NOTES["Weasly Fish"]

    assert "58.69" in notes, "mid-body width"
    assert "29.56" in notes, "caudal peduncle width"


def test_weasly_fish_uses_the_previously_published_length():
    """31 cm, matching a prior publication.

    Deliberately NOT the midpoint of the known [300, 310] mm interval, which
    would halve the worst-case reference error and make it symmetric. A
    reference that disagrees with an already-published number is worse than a
    slightly biased one: it turns every future comparison into something a
    reader has to reconcile by hand.
    """
    weasly = next(m for m in KNOWN_FISH_MODELS if m["name"] == "Weasly Fish")

    assert weasly["known_length_m"] == pytest.approx(0.310)


def test_weasly_fish_records_the_uncertainty_its_length_carries():
    """310 is the TOP of the known interval, so the reference is one-sided.

    A perfect measurement of this model reads 0.00%..-3.23% from the reference
    alone. Without that written down, the first small negative reading gets
    blamed on calibration — which is exactly the mistake the ruler's own
    history is a monument to.
    """
    notes = FISH_MODEL_NOTES["Weasly Fish"]

    assert "length_range_mm=300-310" in notes
    assert "3.23" in notes
    assert "provisional" in notes.lower()


def test_a_perfect_measurement_of_weasly_fish_reads_within_the_stated_band():
    """Pins the arithmetic the note asserts, so the two cannot drift."""
    ref = next(
        m for m in KNOWN_FISH_MODELS if m["name"] == "Weasly Fish"
    )["known_length_m"]

    errors = [100.0 * (truth - ref) / ref for truth in (0.300, 0.310)]

    assert min(errors) == pytest.approx(-3.23, abs=0.01)
    assert max(errors) == pytest.approx(0.0, abs=0.01)


def test_ruler_is_seeded_at_the_labeled_span_not_the_nominal_length():
    """The ruler is a 14-inch ruler, but the LABELED span is 13.5 in.

    Labelers click the first printed graduation — the 0.5 mark, the leftmost
    thing on the scale — and the 14 mark. Measuring the ruler's own inch ticks
    across 4 frames gave 13.500/13.505/13.481/13.468 in (SD 0.13%).

    Pinned as a regression because 14 in is the intuitive-but-wrong value, and
    getting it wrong cost a whole false explanation (an invented 22-degree tilt)
    before anyone checked the head end of the scale.
    """
    ruler = next(m for m in KNOWN_FISH_MODELS if m["name"] == "Ruler")
    assert ruler["known_length_m"] == pytest.approx(0.3429)  # 13.5 in
    assert ruler["known_length_m"] != pytest.approx(0.3556), "14 in is the trap"


def test_known_lengths_are_positive_and_plausible():
    for model in KNOWN_FISH_MODELS:
        assert 0.05 < model["known_length_m"] < 2.0, model


def test_known_model_names_are_unique():
    names = [m["name"] for m in KNOWN_FISH_MODELS]
    assert len(names) == len(set(names))


# ── the reference table ───────────────────────────────────────────────


async def test_reference_row_round_trips(session):
    from fishsense_api.models.fish_model_reference import (
        FishModelReference,
    )

    session.add(FishModelReference(name="Grouper", known_length_m=0.360))
    await session.flush()

    row = (
        await session.exec(
            select(FishModelReference).where(FishModelReference.name == "Grouper")
        )
    ).first()
    assert row is not None
    assert row.known_length_m == pytest.approx(0.360)


# ── the accuracy view ─────────────────────────────────────────────────


async def _seed_measurement(session, *, dive_id, image_id, model_name, length_m):
    from datetime import datetime, timezone

    from fishsense_api.models.dive import Dive
    from fishsense_api.models.fish import Fish
    from fishsense_api.models.image import Image
    from fishsense_api.models.measurement import Measurement
    from fishsense_api.models.priority import Priority

    existing = (await session.exec(select(Dive).where(Dive.id == dive_id))).first()
    if existing is None:
        session.add(
            Dive(
                id=dive_id,
                path=f"/dev/null/{dive_id}",
                dive_datetime=datetime(2023, 8, 29, tzinfo=timezone.utc),
                priority=Priority.HIGH,
            )
        )
        await session.flush()
    fish = (
        await session.exec(select(Fish).where(Fish.name == model_name))
    ).first()
    if fish is None:
        fish = Fish(name=model_name, species_id=None)
        session.add(fish)
        await session.flush()
    session.add(
        Image(
            id=image_id,
            path=f"/dev/null/img-{image_id}",
            taken_datetime=datetime(2023, 8, 29, tzinfo=timezone.utc),
            checksum=f"{image_id:032d}",
            dive_id=dive_id,
        )
    )
    await session.flush()
    session.add(Measurement(image_id=image_id, fish_id=fish.id, length_m=length_m))
    await session.flush()


async def _rows(session):
    result = await session.exec(
        text(
            "SELECT dive_id, model_name, known_length_m, length_m, "
            "error_m, pct_error FROM fish_model_measurement_accuracy "
            "ORDER BY image_id"
        )
    )
    return [dict(r._mapping) for r in result]  # pylint: disable=protected-access


async def test_accuracy_view_computes_error_against_known_length(session):
    from fishsense_api.models.fish_model_reference import (
        FishModelReference,
    )

    session.add(FishModelReference(name="Grouper", known_length_m=0.360))
    await session.flush()
    # Measured 10% long.
    await _seed_measurement(
        session, dive_id=1, image_id=11, model_name="Grouper", length_m=0.396
    )

    rows = await _rows(session)

    assert len(rows) == 1
    assert rows[0]["model_name"] == "Grouper"
    assert rows[0]["known_length_m"] == pytest.approx(0.360)
    assert rows[0]["error_m"] == pytest.approx(0.036, abs=1e-6)
    assert rows[0]["pct_error"] == pytest.approx(10.0, abs=1e-4)


async def test_accuracy_view_excludes_real_fish(session):
    """Real (wild) fish carry name=NULL and have no reference row — they must
    not appear, or the view stops being a model-accuracy view."""
    from datetime import datetime, timezone

    from fishsense_api.models.dive import Dive
    from fishsense_api.models.fish import Fish
    from fishsense_api.models.image import Image
    from fishsense_api.models.measurement import Measurement
    from fishsense_api.models.priority import Priority
    from fishsense_api.models.fish_model_reference import (
        FishModelReference,
    )

    session.add(FishModelReference(name="Grouper", known_length_m=0.360))
    session.add(
        Dive(
            id=1,
            path="/dev/null/1",
            dive_datetime=datetime(2023, 8, 29, tzinfo=timezone.utc),
            priority=Priority.HIGH,
        )
    )
    await session.flush()
    wild = Fish(name=None, species_id=None)
    session.add(wild)
    await session.flush()
    session.add(
        Image(
            id=11,
            path="/dev/null/img-11",
            taken_datetime=datetime(2023, 8, 29, tzinfo=timezone.utc),
            checksum="c" * 32,
            dive_id=1,
        )
    )
    await session.flush()
    session.add(Measurement(image_id=11, fish_id=wild.id, length_m=0.5))
    await session.flush()

    assert await _rows(session) == []


async def test_accuracy_view_excludes_models_without_a_reference(session):
    """A model nobody has measured with calipers yet can't be graded — it must
    be absent rather than silently compared against NULL."""
    await _seed_measurement(
        session, dive_id=1, image_id=11, model_name="Unmeasured Model", length_m=0.4
    )

    assert await _rows(session) == []
