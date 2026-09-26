"""`laserlabel.superseded_reason` records WHY a laser label was superseded.

`superseded` is a dead letter with no provenance. When the laser validator
turned out to have eroded dives a run at a time (14,523 of 47,562 positive
labels superseded by 2026-09-26, most of them good), nothing on the row could
tell a validator flag from a deliberate operator fix — dive 77's reflection
recipe, or the wild slate dots on 347 — so remediation needs a hand-supplied
exclusion list, and the v2 migration can carry nothing better than the bit.

From here on every supersede says who did it. Existing rows stay NULL, which
reads as "unknown": the migration does not guess.

Stored as a plain VARCHAR, not a Postgres native enum, so adding a reason is a
code change rather than an `ALTER TYPE ... ADD VALUE` (the `b3d5e91a7c42`
lesson).
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlmodel import select

from tests_support.db import dive, image

IMAGE_ID = 11
PROJECT_ID = 7


def test_the_vocabulary():
    """The four writers. Pinned so a rename is a deliberate, reviewed act —
    the v2 migration and the remediation tool key on these strings."""
    from fishsense_api.models.superseded_reason import SupersededReason

    assert {reason.value for reason in SupersededReason} == {
        "validator_3sigma",
        "validator_coarse_calibration",
        "manual",
        "remediation",
    }


# --- migration ----------------------------------------------------------------


@pytest.fixture
def migration():
    from fishsense_api.alembic.versions import (
        e5a9c3d71b24_add_laserlabel_superseded_reason as mod,
    )

    return mod


@pytest.fixture
def engine():
    """`laserlabel` as it was, with a superseded row written before the column."""
    eng = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    table = sa.Table(
        "laserlabel",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("superseded", sa.Boolean()),
    )
    metadata.create_all(eng)
    with eng.begin() as conn:
        conn.execute(table.insert().values(id=1, superseded=True))
    return eng


def _run(migration, conn, direction="upgrade"):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    with Operations.context(MigrationContext.configure(conn)):
        getattr(migration, direction)()


def test_the_migration_follows_the_current_head(migration):
    assert migration.down_revision == "c4e8b17a205f"


def test_existing_rows_read_as_unknown(migration, engine):
    """No backfill: a pre-existing supersede's reason is genuinely unknown."""
    with engine.begin() as conn:
        _run(migration, conn)
        col = next(
            c
            for c in sa.inspect(conn).get_columns("laserlabel")
            if c["name"] == "superseded_reason"
        )
        assert col["nullable"] is True
        assert conn.execute(
            sa.text("SELECT superseded_reason FROM laserlabel WHERE id = 1")
        ).scalar() is None


def test_downgrade_drops_the_column(migration, engine):
    with engine.begin() as conn:
        _run(migration, conn)
        _run(migration, conn, "downgrade")
        cols = {c["name"] for c in sa.inspect(conn).get_columns("laserlabel")}
        assert "superseded_reason" not in cols


# --- the API --------------------------------------------------------------------


async def _seed(session):
    session.add(dive(1))
    session.add(image(IMAGE_ID, 1))
    await session.flush()


def _body(**extra):
    return {
        "image_id": IMAGE_ID,
        "label_studio_project_id": PROJECT_ID,
        "label_studio_task_id": 99,
        "x": 10.0,
        "y": 20.0,
        "completed": True,
        **extra,
    }


async def _row(session):
    from fishsense_api.models.laser_label import LaserLabel

    rows = (
        await session.exec(select(LaserLabel).where(LaserLabel.image_id == IMAGE_ID))
    ).all()
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize(
    "reason",
    ["validator_3sigma", "validator_coarse_calibration", "manual", "remediation"],
)
async def test_a_reason_round_trips_over_http(http, session, reason):
    await _seed(session)

    put = http.put(
        f"/api/v1/labels/laser/{IMAGE_ID}",
        json=_body(superseded=True, superseded_reason=reason),
    )
    assert put.status_code == 201, put.text

    got = http.get("/api/v1/dives/1/labels/laser?include_superseded=true")
    assert got.status_code == 200, got.text
    assert [row["superseded_reason"] for row in got.json()] == [reason]


async def test_an_unknown_reason_is_refused(http, session):
    """A typo would otherwise land as a reason nothing downstream recognises."""
    await _seed(session)

    put = http.put(
        f"/api/v1/labels/laser/{IMAGE_ID}",
        json=_body(superseded=True, superseded_reason="validator_3_sigma"),
    )

    assert put.status_code == 422, put.text


async def test_a_put_that_omits_the_reason_keeps_it(http, session):
    """The hourly sync rewrites laser rows without knowing this column exists;
    it must not erase the record of why a row was superseded."""
    await _seed(session)
    first = http.put(
        f"/api/v1/labels/laser/{IMAGE_ID}",
        json=_body(superseded=True, superseded_reason="manual"),
    )
    assert first.status_code == 201, first.text

    again = http.put(f"/api/v1/labels/laser/{IMAGE_ID}", json=_body(label="kp-2"))
    assert again.status_code == 201, again.text

    row = await _row(session)
    assert row.superseded is True
    assert row.superseded_reason == "manual"
