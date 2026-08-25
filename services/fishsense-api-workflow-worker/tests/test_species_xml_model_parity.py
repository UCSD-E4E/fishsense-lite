"""The species XML and `taxonomy.LABELED_FISH_MODELS` must not drift.

`LABELED_FISH_MODELS` is what the API grades against — it decides which models
need a `fishmodelreference` row. The species labeling XML is what a labeler can
actually pick. If the two disagree, the disagreement is **silent in both
directions**:

  * a model in the XML but not the list gets no reference row, and the accuracy
    view inner-joins it away — its measurements simply never appear (this is
    exactly what `Weasly Fish` did);
  * a model in the list but not the XML demands a reference row nobody can ever
    produce a measurement for.

The XML lives in the api-worker and the grading lives in the API, which is why
the shared list exists at all and why this parity test does.
"""

from __future__ import annotations

import re

from fishsense_shared import taxonomy

from fishsense_api_workflow_worker.activities.create_species_label_studio_project_activity import (  # noqa: E501  pylint: disable=line-too-long
    SPECIES_LABELING_CONFIG_XML,
)


def _xml_fish_models() -> list[str]:
    """Names nested under the `Fish Model` parent choice, in XML order.

    Parsed from the real pasted-from-prod XML rather than a copy, so the test
    fails when someone edits the config — which is the moment it matters.
    """
    block = re.search(
        r'<Choice value="Fish Model">(.*?)</Choice>\s*<Choice value="Calibration Targets">',
        SPECIES_LABELING_CONFIG_XML,
        re.S,
    )
    assert block, "Fish Model branch not found — did the XML change shape?"
    return re.findall(r'<Choice value="([^"]+)"\s*/>', block.group(1))


def test_the_parser_actually_finds_the_models():
    """Guards the regex itself: a parse that silently returned [] would make
    every assertion below vacuously true."""
    assert len(_xml_fish_models()) >= 6


def test_every_xml_fish_model_is_in_the_shared_list():
    missing = set(_xml_fish_models()) - set(taxonomy.LABELED_FISH_MODELS)

    assert not missing, f"in the species XML but ungradeable: {sorted(missing)}"


def test_the_shared_list_invents_no_models():
    extra = set(taxonomy.LABELED_FISH_MODELS) - set(_xml_fish_models())

    assert not extra, f"listed but not labelable: {sorted(extra)}"


def test_weasly_fish_is_labelable():
    """The specific row this work exists for — it was pickable in prod with no
    reference row, so its measurements were graded by nobody."""
    assert "Weasly Fish" in _xml_fish_models()
