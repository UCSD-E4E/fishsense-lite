"""Unit tests for the `content_of_image` taxonomy vocabulary.

`SpeciesLabel.content_of_image` is a ", "-joined Label Studio taxonomy path,
and four consumers read it: the stage-14 measure activity (Python), the
stage-14 cohort selector (SQLAlchemy), the `dive_pipeline_status` view (raw
SQL), and the stage-9 slate cohort. They used to spell the markers
independently, and the docstrings describing them had drifted so far that the
worked example in two files said `"Fish Model, …"` and
`"Calibration Targets, Ruler"` were skipped while the code six lines below
matched both as measurable.

`MEASURABILITY_CORPUS` is the shared fixture: every branch that actually
occurs, plus the boundary cases. `test_dive_pipeline_status_view.py` runs the
*SQL* predicate over the same corpus and asserts it agrees with
`is_measurable` here, so the two representations can't drift silently again.
"""

from __future__ import annotations

import pytest

from fishsense_shared import taxonomy as sut


def test_markers_match_the_label_studio_config():
    assert sut.FISH_MODEL_PREFIX == "Fish Model,"
    assert sut.RULER_CONTENT == "Calibration Targets, Ruler"
    assert sut.RULER_NAME == "Ruler"
    assert sut.SLATE_CONTENT_MARKER == "Slate, Laser on slate"


# --------------------------------------------------------------------
# parse_species_names — real (wild) fish
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            "Fish, Hogfish (Lachnolaimus maximus)",
            ("Hogfish", "Lachnolaimus maximus"),
        ),
        (
            "Fish, Stoplight Parrotfish (Sparisoma viride)",
            ("Stoplight Parrotfish", "Sparisoma viride"),
        ),
        # Only the LAST ", "-chunk is the species; earlier chunks are the
        # taxonomy path the labeler drilled through.
        ("Fish, Reef, Bar Jack (Caranx ruber)", ("Bar Jack", "Caranx ruber")),
    ],
)
def test_parse_species_names_reads_the_last_chunk(content, expected):
    assert sut.parse_species_names(content) == expected


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "   ",
        "Fish Model, Weasly Fish",  # no parens
        "Calibration Targets, Ruler",  # no parens
        "Slate, Laser on slate",  # no parens
        "Fish, Hogfish (",  # unbalanced
        "Fish, Hogfish )",  # no opening paren
        "Fish,  (Lachnolaimus maximus)",  # empty common name
        "Fish, Hogfish ()",  # empty scientific name
    ],
)
def test_parse_species_names_returns_none_off_shape(content):
    """We skip rather than write a malformed Species row."""
    assert sut.parse_species_names(content) is None


# --------------------------------------------------------------------
# parse_model_name — rigid known-length targets
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("Fish Model, Weasly Fish", "Weasly Fish"),
        ("Fish Model, Snook", "Snook"),
        ("Fish Model, Purple Angel", "Purple Angel"),
        # The ruler is a rigid known-length target like the models, so it
        # resolves through the same name-keyed path.
        ("Calibration Targets, Ruler", "Ruler"),
    ],
)
def test_parse_model_name(content, expected):
    assert sut.parse_model_name(content) == expected


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "Fish Model,",  # empty leaf — nothing to identify
        "Fish Model,   ",
        "Fish, Hogfish (Lachnolaimus maximus)",  # a real fish
        "Calibration Targets, Slate",  # not the ruler
        "Slate, Laser on slate",
    ],
)
def test_parse_model_name_returns_none_off_shape(content):
    assert sut.parse_model_name(content) is None


def test_a_value_is_never_both_a_species_and_a_model():
    """`measure_fish_activity` computes both and branches on which is set, so
    an overlap would make the binding order-dependent."""
    for content, _ in sut.MEASURABILITY_CORPUS:
        both = (
            sut.parse_species_names(content) is not None
            and sut.parse_model_name(content) is not None
        )
        assert not both, f"{content!r} parses as both a species and a model"


# --------------------------------------------------------------------
# is_measurable
# --------------------------------------------------------------------


@pytest.mark.parametrize(("content", "expected"), sut.MEASURABILITY_CORPUS)
def test_is_measurable_over_the_corpus(content, expected):
    assert sut.is_measurable(content) is expected


def test_is_measurable_is_exactly_what_the_activity_can_bind():
    """The definition that matters: measurable == the activity will produce a
    Measurement. Any looser and the stage-14 cohort offers an image the
    activity always skips, no Measurement is ever written, and the dive is
    re-selected every hour forever — the never-goes-false shape that blocked
    scheduling stage 14 before 2026-07-17."""
    for content, _ in sut.MEASURABILITY_CORPUS:
        bindable = (
            sut.parse_species_names(content) is not None
            or sut.parse_model_name(content) is not None
        )
        assert sut.is_measurable(content) is bindable


def test_corpus_covers_both_measurable_and_unmeasurable():
    """A corpus that drifted to all-True would make the SQL parity test
    vacuous."""
    outcomes = {expected for _, expected in sut.MEASURABILITY_CORPUS}
    assert outcomes == {True, False}


# --------------------------------------------------------------------
# SQL fragment builders
# --------------------------------------------------------------------


def test_rigid_target_sql_excludes_the_empty_leaf():
    """The guard that stops a labeler mis-click wedging the stage-14 cohort:
    `LIKE 'Fish Model,%'` alone matches `"Fish Model,"`, which
    `parse_model_name` rejects."""
    sql = sut.rigid_target_sql("sl.content_of_image")
    assert "LIKE 'Fish Model,%'" in sql
    assert "TRIM(sl.content_of_image) <> 'Fish Model,'" in sql
    assert "sl.content_of_image = 'Calibration Targets, Ruler'" in sql


def test_measurable_species_sql_is_real_fish_or_rigid_target():
    sql = sut.measurable_species_sql("sl.content_of_image")
    assert "LIKE '%(%)'" in sql
    assert sut.rigid_target_sql("sl.content_of_image") in sql


def test_sql_builders_take_the_column_name():
    """The view aliases specieslabel as `sl`; a caller with a different alias
    must not have to string-replace."""
    assert "x.content" in sut.measurable_species_sql("x.content")
    assert "sl.content_of_image" not in sut.measurable_species_sql("x.content")


def test_sql_broader_rows_are_unmeasurable_in_python():
    """The pinned divergence set must actually be Python-unmeasurable — that
    is the whole claim being tracked."""
    for content in sut.SQL_BROADER_THAN_PYTHON:
        assert sut.is_measurable(content) is False


def test_sql_broader_rows_are_not_also_in_the_corpus():
    """The corpus asserts exact agreement; the divergence tuple asserts a known
    mismatch. A value in both would make one of the two tests a lie."""
    corpus_values = {c for c, _ in sut.MEASURABILITY_CORPUS}
    assert not corpus_values & set(sut.SQL_BROADER_THAN_PYTHON)
