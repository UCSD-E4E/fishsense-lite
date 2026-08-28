"""The stage version is a judgement call, pinned where it can be.

`LASER_PREDICTOR_VERSION` has to be bumped by hand whenever the laser-detector
stage's output would differ for an unchanged image. No test can catch every
such change — that is exactly why it is a literal in a diff rather than a hash
computed at runtime — but the inputs that are cheap to pin are pinned here, so
the most likely silent change (editing the region the stage gates on, without
bumping) fails CI instead of shipping.
"""

from __future__ import annotations

import pytest

from fishsense_shared import LASER_PREDICTOR_VERSION, laser_model_version_tag
from fishsense_shared.laser_region import LASER_REGION_POLYGON


def test_version_is_a_positive_int():
    assert isinstance(LASER_PREDICTOR_VERSION, int)
    assert LASER_PREDICTOR_VERSION >= 1


def test_the_region_the_stage_gates_on_is_pinned_to_this_version():
    """The region is an input to the stage's output: change it and predictions
    that used to be accepted are rejected. Bump the version in the same commit
    and update this fixture -- it is here so that cannot be forgotten."""
    assert LASER_REGION_POLYGON == [
        [1580, 570],
        [1700, 465],
        [2335, 395],
        [2455, 525],
        [2470, 1610],
        [2185, 1890],
        [1920, 1905],
        [1625, 1365],
    ], "the laser region changed -- bump LASER_PREDICTOR_VERSION and this fixture"
    assert LASER_PREDICTOR_VERSION == 2


def test_the_ls_tag_carries_the_version():
    """Both the pre-annotation a labeler sees and the backfill's idempotency
    check key on this string. If it stopped varying with the version, a task
    seeded by an older stage would look current and never be refreshed."""
    assert laser_model_version_tag() == f"laser-detector-v{LASER_PREDICTOR_VERSION}"
    assert laser_model_version_tag(1) != laser_model_version_tag(2)


def test_the_tag_is_not_the_old_bare_constant():
    """Guards the regression: it was "laser-detector" for every version."""
    assert laser_model_version_tag() != "laser-detector"


@pytest.mark.parametrize("version", [1, 2, 17])
def test_tag_is_stable_for_a_given_version(version):
    assert laser_model_version_tag(version) == laser_model_version_tag(version)
