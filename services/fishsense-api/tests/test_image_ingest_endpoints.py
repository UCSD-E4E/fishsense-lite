"""Tests for `POST /api/v1/dives/{dive_id}/images/` and the batch checksum
lookup — the image half of ingest.

`is_canonical` is the subtle one. The same physical frames legitimately appear
under two dive rows (prod dives 64 and 66 are both `082929_FishModels_FSL07`,
55 images each), and `checksum` is how that is recognised. The rule comes from
commit `9e5bc64`:

    is_canonical = (existing_checksum is None)

i.e. the first row for a given checksum is canonical, later duplicates are not.
It is computed **server-side** here rather than by the caller: a client-side
"does this checksum exist yet" check races itself, and two concurrent posts
would both decide they were first.

The batch lookup backs duplicate-dive detection (plan §4.2.1) — the replacement
for spider's whole-dive MD5 aggregate, which was all-or-nothing and
basename-sensitive.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession


@pytest.fixture
async def session():
    import fishsense_api.database  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _seed_dive(session, dive_id: int, path: str):
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    session.add(
        Dive(
            id=dive_id,
            path=path,
            dive_datetime=datetime(2024, 8, 21, 8, 56, 51, tzinfo=timezone.utc),
            priority=Priority.LOW,
            camera_id=1,
        )
    )
    await session.flush()


def _image(path: str, checksum: str, **kwargs):
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    kwargs.setdefault("camera_id", 1)
    return Image(
        path=path,
        checksum=checksum,
        taken_datetime=datetime(2024, 8, 21, 8, 56, 51, tzinfo=timezone.utc),
        **kwargs,
    )


CK_A = "45dc5a454b35601b9dafabf24822195d"
CK_B = "0b7cd4da72d54172f1f9daf40ce4047f"


# ── creation ──────────────────────────────────────────────────────────


async def test_post_image_creates_a_row_and_binds_it_to_the_path_dive(session):
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        post_image,
    )
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    await _seed_dive(session, 7, "d/7")

    # dive_id in the body is deliberately wrong — the path must win.
    image_id = await post_image(
        7, _image("d/7/P8210001.ORF", CK_A, dive_id=999), session=session
    )

    row = (await session.exec(select(Image).where(Image.id == image_id))).first()
    assert row is not None
    assert row.dive_id == 7
    assert row.checksum == CK_A


async def test_post_image_is_an_upsert_on_path_not_a_duplicate_insert(session):
    """`Image.path` is unique. Resuming a partial scan re-posts paths that are
    already there; a blind merge would 500 on the unique index."""
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        post_image,
    )
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    await _seed_dive(session, 7, "d/7")
    path = "d/7/P8210001.ORF"

    first = await post_image(7, _image(path, CK_A), session=session)
    second = await post_image(7, _image(path, CK_A), session=session)

    assert first == second
    rows = (await session.exec(select(Image).where(Image.path == path))).all()
    assert len(rows) == 1


# ── is_canonical ──────────────────────────────────────────────────────


async def test_first_image_with_a_checksum_is_canonical_later_duplicates_are_not(session):
    """The dives-64/66 rule, from `9e5bc64`."""
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        post_image,
    )
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    await _seed_dive(session, 64, "d/64")
    await _seed_dive(session, 66, "d/66")

    first = await post_image(64, _image("d/64/P1.ORF", CK_A), session=session)
    dupe = await post_image(66, _image("d/66/P1.ORF", CK_A), session=session)

    rows = {
        r.id: r
        for r in (await session.exec(select(Image))).all()
    }
    assert rows[first].is_canonical is True
    assert rows[dupe].is_canonical is False


async def test_reposting_the_same_path_does_not_demote_it_to_non_canonical(session):
    """A resumed scan re-posts an existing path. The row it would collide with
    is *itself*, so 'a row with this checksum already exists' must not flip it
    to False — that would quietly strip canonical status from a whole dive on
    every re-run."""
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        post_image,
    )
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    await _seed_dive(session, 7, "d/7")
    path = "d/7/P8210001.ORF"

    await post_image(7, _image(path, CK_A), session=session)
    image_id = await post_image(7, _image(path, CK_A), session=session)

    row = (await session.exec(select(Image).where(Image.id == image_id))).first()
    assert row.is_canonical is True


async def test_an_explicit_is_canonical_in_the_body_overrides_the_computation(session):
    """Operator override — e.g. promoting a re-ingested copy after the original
    dive's files were lost."""
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        post_image,
    )
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    await _seed_dive(session, 64, "d/64")
    await _seed_dive(session, 66, "d/66")

    await post_image(64, _image("d/64/P1.ORF", CK_A), session=session)
    dupe = await post_image(
        66, _image("d/66/P1.ORF", CK_A, is_canonical=True), session=session
    )

    row = (await session.exec(select(Image).where(Image.id == dupe))).first()
    assert row.is_canonical is True


# ── validation ────────────────────────────────────────────────────────


async def test_post_image_rejects_a_path_longer_than_the_column(session):
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        post_image,
    )

    await _seed_dive(session, 7, "d/7")

    with pytest.raises(HTTPException) as exc:
        await post_image(7, _image("x" * 256, CK_A), session=session)
    assert exc.value.status_code == 422


async def test_post_image_rejects_a_checksum_that_is_not_a_32_char_md5(session):
    """A wrong-width checksum means the hashing changed underneath us; every
    duplicate check downstream would silently stop matching."""
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        post_image,
    )

    await _seed_dive(session, 7, "d/7")

    for bad in ("", "deadbeef", "z" * 32, "a" * 33):
        with pytest.raises(HTTPException) as exc:
            await post_image(7, _image(f"d/7/{bad or 'empty'}.ORF", bad), session=session)
        assert exc.value.status_code == 422, bad


async def test_post_image_rejects_an_unknown_dive(session):
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        post_image,
    )

    with pytest.raises(HTTPException) as exc:
        await post_image(4242, _image("d/x/P1.ORF", CK_A), session=session)
    assert exc.value.status_code == 422


# ── batch checksum lookup (plan §4.2.1) ───────────────────────────────


async def test_checksum_lookup_reports_every_dive_holding_each_checksum(session):
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        post_checksum_lookup,
        post_image,
    )

    await _seed_dive(session, 64, "d/64")
    await _seed_dive(session, 66, "d/66")
    await post_image(64, _image("d/64/P1.ORF", CK_A), session=session)
    await post_image(66, _image("d/66/P1.ORF", CK_A), session=session)
    await post_image(64, _image("d/64/P2.ORF", CK_B), session=session)

    result = await post_checksum_lookup([CK_A, CK_B, "f" * 32], session=session)

    assert {hit["dive_id"] for hit in result[CK_A]} == {64, 66}
    assert {hit["dive_id"] for hit in result[CK_B]} == {64}
    # Canonicality travels with the hit so the caller can explain the
    # consequence ("these will land non-canonical") without a second query.
    assert {hit["is_canonical"] for hit in result[CK_A]} == {True, False}


async def test_checksum_lookup_returns_an_empty_list_for_unknown_checksums(session):
    """Empty list, not a missing key — the caller computes
    `|new n existing| / |new|` and should not have to guard every lookup."""
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        post_checksum_lookup,
    )

    result = await post_checksum_lookup([CK_A], session=session)

    assert result == {CK_A: []}


async def test_checksum_lookup_is_a_set_operation_not_an_ordered_digest(session):
    """The property spider's whole-dive MD5 aggregate lacked (plan SS4.2.1).

    A folder sharing 2 of 3 frames with an existing dive must report partial
    overlap. The aggregate reported such a folder as simply *different*, which
    is why it never worked well: near-duplicates are the common case, and
    filename order changed the digest even when the bytes matched.
    """
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        post_checksum_lookup,
        post_image,
    )

    await _seed_dive(session, 64, "d/64")
    await post_image(64, _image("d/64/P1.ORF", CK_A), session=session)
    await post_image(64, _image("d/64/P2.ORF", CK_B), session=session)

    incoming = [CK_A, CK_B, "e" * 32]
    result = await post_checksum_lookup(incoming, session=session)

    shared = [ck for ck in incoming if any(h["dive_id"] == 64 for h in result[ck])]
    assert len(shared) / len(incoming) == pytest.approx(2 / 3)


# ── partial update semantics + remaining guards ───────────────────────


async def test_reposting_an_image_preserves_fields_the_body_did_not_mention(session):
    """Same destructive-upsert class as the dive endpoint. A resumed scan
    re-posts paths it has already written; anything it omits must survive."""
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        post_image,
    )
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    await _seed_dive(session, 7, "d/7")
    path = "d/7/P8210001.ORF"
    image_id = await post_image(7, _image(path, CK_A, camera_id=3), session=session)

    await post_image(
        7,
        Image(
            path=path,
            checksum=CK_A,
            taken_datetime=datetime(2024, 8, 21, 8, 56, 51, tzinfo=timezone.utc),
        ),
        session=session,
    )

    row = (await session.exec(select(Image).where(Image.id == image_id))).first()
    assert row.camera_id == 3


async def test_post_image_rejects_an_empty_path(session):
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        post_image,
    )

    await _seed_dive(session, 7, "d/7")

    with pytest.raises(HTTPException) as exc:
        await post_image(7, _image("", CK_A), session=session)
    assert exc.value.status_code == 422


async def test_post_image_rejects_a_missing_taken_datetime(session):
    """Stage-1 clustering is pure timestamp math, so a defaulted or absent
    `taken_datetime` would corrupt it with nothing to show for it."""
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        post_image,
    )
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    await _seed_dive(session, 7, "d/7")
    no_date = Image(path="d/7/P1.ORF", checksum=CK_A, taken_datetime=None)

    with pytest.raises(HTTPException) as exc:
        await post_image(7, no_date, session=session)
    assert exc.value.status_code == 422


async def test_checksum_lookup_rejects_an_oversized_batch(session):
    """Unbounded `IN (...)` from a bad caller."""
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        MAX_CHECKSUM_LOOKUP,
        post_checksum_lookup,
    )

    too_many = [f"{i:032x}" for i in range(MAX_CHECKSUM_LOOKUP + 1)]

    with pytest.raises(HTTPException) as exc:
        await post_checksum_lookup(too_many, session=session)
    assert exc.value.status_code == 422


async def test_checksum_lookup_handles_an_empty_batch(session):
    """A dive folder that turned out to hold nothing new still calls this."""
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        post_checksum_lookup,
    )

    assert await post_checksum_lookup([], session=session) == {}


async def test_checksum_lookup_deduplicates_repeated_checksums(session):
    """Duplicate frames inside one folder are exactly the case this is for, so
    the same checksum arriving twice must not double-count toward the cap or
    produce duplicate hits."""
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        post_checksum_lookup,
        post_image,
    )

    await _seed_dive(session, 64, "d/64")
    await post_image(64, _image("d/64/P1.ORF", CK_A), session=session)

    result = await post_checksum_lookup([CK_A, CK_A, CK_A], session=session)

    assert list(result) == [CK_A]
    assert len(result[CK_A]) == 1
