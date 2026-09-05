"""The migration graph is walkable, with exactly one head.

This exists because of a real outage-shaped miss. `c4f8a2e60b17` shipped with
`down_revision = "e9c1a7b40d53"` — a revision that lived only on an unmerged
branch, picked up because the head was read off a working tree that happened to
contain someone else's in-progress migration. Alembic could not resolve it, and
`run_alembic_upgrade` raises `KeyError` on startup, so the API does not come up.

It was caught, but only by `test_api_postgres_integration.py`, which walks the
chain for real against Postgres — and that suite is **deselected on ordinary
PRs** (it needs `workflow_dispatch`, the `integration-tests` label, or the
release-please branch). So the PR went green with eleven latent failures in it
and the break surfaced after merge, on main, where it had to be fixed forward
(#696).

The check therefore has to live in the tier that always runs. These tests need
no database and no stack: they read the scripts off disk, which is all that is
required to catch a dangling parent or a fork. Keep them fast and dependency-
free for that reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory


@pytest.fixture(name="scripts")
def _scripts() -> ScriptDirectory:
    """The migration directory, located the same way `run_alembic_upgrade`
    locates it — relative to the package, because `alembic.ini` does not ship
    inside the wheel."""
    import fishsense_api

    cfg = AlembicConfig()
    cfg.set_main_option(
        "script_location",
        str(Path(fishsense_api.__file__).resolve().parent / "alembic"),
    )
    return ScriptDirectory.from_config(cfg)


def test_there_is_exactly_one_head(scripts: ScriptDirectory):
    """Two heads mean two migrations claim to be last, which is what happens
    when branches add migrations in parallel and neither rebases. `upgrade
    head` then refuses to pick."""
    heads = scripts.get_heads()
    assert len(heads) == 1, (
        f"expected a single alembic head, found {len(heads)}: {heads}. "
        "Two branches added migrations without chaining; set the later one's "
        "down_revision to the other."
    )


def test_every_revision_resolves_from_head_to_base(scripts: ScriptDirectory):
    """Walk the whole graph the way `upgrade head` does.

    A `down_revision` naming a revision that is not on disk raises here with
    the offending id, which is the exact failure `c4f8a2e60b17` shipped.
    """
    walked = list(scripts.walk_revisions("base", "heads"))
    assert walked, "no migrations found — is script_location wrong?"


def test_no_migration_points_at_a_revision_that_does_not_exist(
    scripts: ScriptDirectory,
):
    """Same property as above, but reported per-file rather than as whichever
    one alembic trips over first — so a reviewer sees every dangling parent at
    once instead of fixing them one CI run at a time."""
    known = {revision.revision for revision in scripts.walk_revisions("base", "heads")}
    dangling = []
    for revision in scripts.walk_revisions("base", "heads"):
        parents = revision.down_revision
        if parents is None:
            continue
        if isinstance(parents, str):
            parents = (parents,)
        dangling.extend(
            f"{revision.revision} -> {parent}"
            for parent in parents
            if parent not in known
        )
    assert not dangling, f"down_revision points at unknown revisions: {dangling}"
