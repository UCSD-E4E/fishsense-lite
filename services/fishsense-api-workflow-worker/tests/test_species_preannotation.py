"""Round-trip pins for species pre-annotations.

The CSV of hand-labelled species judgements (3,000 FSL-01 frames, 2023-08 to
2024-06) is loaded into prod as *sentinel* `SpeciesLabel` rows —
`label_studio_project_id IS NULL`, which every preprocess cohort treats as
"no label", so the dives stay in the labelling flow. This module turns such a
row into a Label Studio prediction so the labeler sees the judgement
pre-filled and confirms it rather than retyping it.

**The test that matters is the round trip.** `build_prediction` is the exact
inverse of the sync activity's parser, and the parser is the definition of
record for what a result shape means. Pinning them against each other is the
only way the two halves cannot drift — the same reasoning as the
taxonomy SQL/Python parity test. A pre-annotation whose shape the parser reads
differently would silently write the wrong column when the labeler accepts it.
"""

from types import SimpleNamespace

import pytest

from fishsense_api_workflow_worker.activities.species_preannotation import (
    build_prediction,
    PREANNOTATION_MODEL_VERSION,
)
from fishsense_api_workflow_worker.activities.sync_species_labels_for_label_studio_project_activity import (  # noqa: E501
    _parse_results,
)


def _row(**kw) -> SimpleNamespace:
    """`build_prediction` reads four attributes, so the tests pass four rather
    than constructing a twenty-field SDK model. The real `SpeciesLabel` carries
    the same four -- `test_the_real_sdk_model_satisfies_the_shape` is the
    tripwire that says so."""
    fields = dict(
        content_of_image=None,
        fish_measurable_category=None,
        fish_angle_category=None,
        fish_curved_category=None,
        grouping=None,
        top_three_photos_of_group=None,
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


# --- the round trip ---------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "Fish, Hogfish (Lachnolaimus maximus)",
        "Fish, Grey Snapper (Lutjanus griseus)",
        "Fish, Other (Identifiable but Nontarget)",
        "Fish, Unidentifiable (Cannot see)",
        "Slate, Laser on slate",
        "Fish Model, Weasly Fish",
        "Calibration Targets, Ruler",
    ],
)
def test_the_species_path_survives_the_round_trip(content):
    pred = build_prediction(_row(content_of_image=content))
    assert _parse_results(pred)["content_of_image"] == content


@pytest.mark.parametrize(
    "measurable", ["yes, center of fish", "yes, not center of fish", "no"]
)
def test_the_measurable_category_survives_the_round_trip(measurable):
    pred = build_prediction(_row(fish_measurable_category=measurable))
    parsed = _parse_results(pred)
    assert parsed["fish_measurable_category"] == measurable


@pytest.mark.parametrize("angle", ["x < 5°", "5° < x < 10°", "10° < x < 15°", "x > 15°"])
def test_the_angle_category_survives_the_round_trip(angle):
    pred = build_prediction(_row(fish_angle_category=angle))
    assert _parse_results(pred)["fish_angle_category"] == angle


@pytest.mark.parametrize("curve", ["No Curve", "Slight Curve", "Significant Curve"])
def test_the_curve_category_survives_the_round_trip(curve):
    pred = build_prediction(_row(fish_curved_category=curve))
    assert _parse_results(pred)["fish_curved_category"] == curve


def test_every_field_survives_together():
    row = _row(
        content_of_image="Fish, Stoplight Parrotfish (Sparisoma viride)",
        fish_measurable_category="yes, not center of fish",
        fish_angle_category="10° < x < 15°",
        fish_curved_category="Slight Curve",
    )
    parsed = _parse_results(build_prediction(row))
    assert parsed["content_of_image"] == row.content_of_image
    assert parsed["fish_measurable_category"] == row.fish_measurable_category
    assert parsed["fish_angle_category"] == row.fish_angle_category
    assert parsed["fish_curved_category"] == row.fish_curved_category


# --- what it deliberately does not emit -------------------------------------


def test_grouping_and_top_three_are_never_pre_annotated():
    """The CSV has no column for either, and they are not guessable: `grouping`
    is a judgement about the frame before this one, and `top_three` selects
    which frames of an individual get measured. Emitting a default would put a
    fabricated answer in front of the labeler, and `top_three` is what stage 14
    keys measurability on — so a wrong one silently decides what gets measured.
    """
    row = _row(
        content_of_image="Fish, Hogfish (Lachnolaimus maximus)",
        grouping="Part of previous group",
        top_three_photos_of_group=True,
    )
    parsed = _parse_results(build_prediction(row))
    assert parsed["grouping"] is None
    assert parsed["top_three_photos_of_group"] is None
    assert all(r["from_name"] not in ("grouping", "exclude") for r in build_prediction(row)["result"])


def test_a_row_with_nothing_to_say_yields_no_prediction():
    assert build_prediction(_row()) is None


def test_the_prediction_is_attributed_to_the_import_not_a_model():
    pred = build_prediction(_row(content_of_image="Fish, Hogfish (Lachnolaimus maximus)"))
    assert pred["model_version"] == PREANNOTATION_MODEL_VERSION
    assert "csv" in PREANNOTATION_MODEL_VERSION.lower()


def test_results_target_the_image_control():
    """`toName="image"` in the labeling config. A mismatched to_name makes LS
    drop the prediction without an error."""
    row = _row(
        content_of_image="Fish, Hogfish (Lachnolaimus maximus)",
        fish_measurable_category="no",
    )
    assert all(r["to_name"] == "image" for r in build_prediction(row)["result"])
    assert {r["from_name"] for r in build_prediction(row)["result"]} == {
        "species",
        "measurable",
    }


def test_the_real_sdk_model_satisfies_the_shape():
    """The tests above use a SimpleNamespace; this is what says the production
    model carries the same four attributes, so the stand-in cannot drift."""
    from fishsense_api_sdk.models.species_label import SpeciesLabel

    for field in (
        "content_of_image",
        "fish_measurable_category",
        "fish_angle_category",
        "fish_curved_category",
    ):
        assert field in SpeciesLabel.model_fields
