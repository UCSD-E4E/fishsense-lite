"""In-memory sqlite session and row builders shared by the controller tests.

Every controller test needs the same three things — a fresh SQLModel schema on
in-memory sqlite, a HIGH-priority dive, and a canonical image on it — and they
each grew their own copy. `duplicate-code` flagged ten such pairs the moment a
tree-wide change made those files move together; six of them reproduce on
untouched main, so the duplication was real and simply invisible to a
changed-files lint run.

FK-less by design: these tests exercise query composition and write semantics,
not referential integrity. Postgres enforces the foreign keys sqlite ignores —
see `test_api_postgres_integration.py`, whose seed needs a real `camera` row.

These are plain functions, called explicitly. The in-memory `session` fixture
lives in `tests/conftest.py` instead, so pytest injects it by name and no test
module has to import a fixture it never calls.
"""

from __future__ import annotations

from datetime import datetime, timezone

# One instant for every seeded row. Tests that care about ordering pass their
# own; the rest only need a valid timestamp.
SEED_DATETIME = datetime(2025, 1, 1, tzinfo=timezone.utc)

__all__ = ["SEED_DATETIME", "dive", "image"]


def dive(
    dive_id: int,
    *,
    priority=None,
    calibration_dive_id: int | None = None,
    dive_slate_id: int | None = None,
    name: str | None = None,
):
    """A HIGH-priority dive, which is what every cohort selector filters on."""
    from fishsense_api.models.dive import Dive
    from fishsense_api.models.priority import Priority

    return Dive(
        id=dive_id,
        path=f"/dev/null/{dive_id}",
        dive_datetime=SEED_DATETIME,
        priority=Priority.HIGH if priority is None else priority,
        calibration_dive_id=calibration_dive_id,
        dive_slate_id=dive_slate_id,
        name=name,
    )


def image(image_id: int, dive_id: int, *, is_canonical: bool = True):
    """Canonical by default — the normal case.

    `Image.is_canonical` defaults to False on the model, which made every
    seeded image a duplicate; harmless while nothing read the flag, misleading
    now that the cohort selectors gate on it.
    """
    from fishsense_api.models.image import Image

    return Image(
        id=image_id,
        path=f"/dev/null/img-{image_id}",
        taken_datetime=SEED_DATETIME,
        checksum=f"{image_id:032d}",
        is_canonical=is_canonical,
        dive_id=dive_id,
    )
