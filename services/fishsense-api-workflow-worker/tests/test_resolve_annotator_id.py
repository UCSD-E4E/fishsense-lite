"""Annotator-id extraction across Label Studio instance shapes.

Self-hosted LS returned `task.annotators` as a list of ints; hosted LS
(app.heartex.com) returns a list of dicts. Every sync activity did
`annotators[-1]` and handed the result to `get_by_label_studio_id`, so on
hosted LS a dict went into the URL path
(`/api/v1/users/label-studio/%7B`) and came back 422 — which escaped the
per-task TaskGroup and failed the entire project's sync.
"""

from __future__ import annotations

import pytest

from fishsense_api_workflow_worker.activities.utils import (
    resolve_annotator_label_studio_id as resolve,
)

_HOSTED = {
    "user_id": 141592,
    "annotated": True,
    "id": 141592,
    "username": "ccrutchf",
    "email": "ccrutchf@ucsd.edu",
}


def test_hosted_ls_dict_yields_the_user_id():
    """The prod shape that caused the 422."""
    assert resolve([_HOSTED]) == 141592


def test_self_hosted_int_list_still_works():
    """Accepted too, so a rollback or mixed instance doesn't break."""
    assert resolve([7, 42]) == 42


def test_takes_the_most_recent_annotator():
    assert resolve([{"user_id": 1}, {"user_id": 2}]) == 2


@pytest.mark.parametrize(
    "annotators",
    [None, [], [{}], [{"username": "no-id"}], ["not-a-number"], [None]],
    ids=["none", "empty", "empty-dict", "no-id-key", "non-numeric", "null-entry"],
)
def test_unusable_shapes_yield_none_rather_than_raising(annotators):
    """None means "skip attribution" — never a value that lands in a URL."""
    assert resolve(annotators) is None


def test_numeric_strings_are_accepted():
    assert resolve(["141592"]) == 141592
    assert resolve([{"user_id": "141592"}]) == 141592


def test_bool_is_not_treated_as_an_id():
    """bool subclasses int; True must not become user 1."""
    assert resolve([True]) is None
