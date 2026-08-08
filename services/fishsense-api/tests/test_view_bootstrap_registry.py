"""`views.ALL_VIEW_DDL` must list every view, or fresh environments lose it.

Views are raw SQL owned by migrations and are not part of
`SQLModel.metadata`. On a fresh database `run_alembic_upgrade` stamps head
rather than upgrading, so the migrations that create them never run — and the
stamped version means it never self-heals. `database._create_all_views` covers
that by replaying `ALL_VIEW_DDL`.

Which makes that tuple a registry with the same failure mode as
`controllers/__init__.py` and `database.py`'s model imports: forget to add
your entry and nothing complains, it just silently isn't there. This test is
the complaint.
"""

from __future__ import annotations

import re

from fishsense_api import views


def _declared_view_names() -> set[str]:
    """Every `*_VIEW_NAME` constant declared in views.py."""
    return {
        getattr(views, name)
        for name in dir(views)
        if name.endswith("_VIEW_NAME")
    }


def test_every_declared_view_is_in_the_bootstrap_registry():
    missing = _declared_view_names() - set(views.ALL_VIEW_NAMES)
    assert not missing, (
        f"{sorted(missing)} declared in views.py but absent from ALL_VIEW_NAMES; "
        "a fresh database would silently not have it"
    )


def test_the_registry_has_one_ddl_pair_per_name():
    assert len(views.ALL_VIEW_DDL) == len(views.ALL_VIEW_NAMES)


def test_every_registry_entry_drops_then_creates_the_same_view():
    """A mismatched pair would drop one view and create another, which reads
    as working right up until the wrong one goes missing."""
    for (drop_sql, create_sql), name in zip(
        views.ALL_VIEW_DDL, views.ALL_VIEW_NAMES, strict=True
    ):
        assert re.search(rf"DROP VIEW IF EXISTS {name}\b", drop_sql), (drop_sql, name)
        assert re.search(rf"CREATE VIEW {name}\b", create_sql), name


def test_registry_names_are_unique():
    assert len(set(views.ALL_VIEW_NAMES)) == len(views.ALL_VIEW_NAMES)
