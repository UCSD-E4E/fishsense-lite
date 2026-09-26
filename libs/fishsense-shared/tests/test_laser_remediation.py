"""The laser-remediation contract between the CLI, the api-worker parent and
the data-worker child.

The digest is what makes apply safe: the child recomputes the plan and refuses
unless its digest equals the one in the report a human reviewed. So it must
name exactly the revivals — nothing else in the report may move it — and must
not depend on the order dives or ids happen to arrive in.
"""

from __future__ import annotations

from fishsense_shared.laser_remediation import (
    RemediateLaserSupersedesInput,
    revival_digest,
)


def test_the_digest_ignores_order():
    assert revival_digest([(8, [3, 1]), (7, [5])]) == revival_digest(
        [(7, [5]), (8, [1, 3])]
    )


def test_the_digest_ignores_dives_with_nothing_to_revive():
    assert revival_digest([(7, [5]), (9, [])]) == revival_digest([(7, [5])])


def test_the_digest_changes_with_any_revival():
    base = revival_digest([(7, [5])])
    assert revival_digest([(7, [5, 6])]) != base
    assert revival_digest([(8, [5])]) != base
    assert revival_digest([]) != base


def test_dry_run_is_the_default():
    request = RemediateLaserSupersedesInput(dive_ids=[7])
    assert request.apply is False
    assert request.expected_plan_sha256 is None
