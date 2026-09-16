"""Turn a stored species judgement into a Label Studio pre-annotation.

A batch of hand-labelled species judgements arrived as a spreadsheet — 3,000
FSL-01 frames spanning 2023-08 to 2024-06, keyed by image checksum. They are
loaded as **sentinel** `SpeciesLabel` rows (`label_studio_project_id IS NULL`),
which every preprocess cohort reads as "no label for this image", so the dives
they belong to stay in the normal labelling flow. This module renders such a
row as an LS prediction, so the labeler is shown the judgement pre-filled and
confirms it instead of retyping it.

**It is the exact inverse of `_parse_results` in the species sync activity**,
and that parser is the definition of record for what a result shape means. The
round-trip is pinned in `test_species_preannotation.py`, for the same reason
the taxonomy SQL/Python parity test exists: a pre-annotation whose shape the
parser reads differently would write the wrong column the moment a labeler
accepted it, with nothing raising.

Three fields are deliberately **not** pre-annotated, and this is the important
part of the design:

* `grouping` — a judgement about the frame *before* this one. The spreadsheet
  has no column for it and it is not inferable from one frame.
* `exclude` / `top_three_photos_of_group` — selects which frames of an
  individual get measured, and stage 14 keys measurability on it. A fabricated
  default would silently decide what gets measured.
* the slate branch's type choice (`H-Slate`, `V-Slate 2`, …) — the spreadsheet
  records a bare `Slate`, which is the taxonomy's parent node, not a leaf.
  `Dive.dive_slate_id` is written only from that choice and a wrong slate
  yields a wrong *scale*, which is precisely what reprojection residual cannot
  see. So the 346 slate rows carry no usable `content_of_image` and are not
  loaded at all; they stay a human's job.
"""

from typing import Any, Dict, Optional, Protocol

#: Stamped on every prediction so a labeler (and anyone reading the task later)
#: can tell this came from the spreadsheet import and not from a model.
PREANNOTATION_MODEL_VERSION = "species-csv-import-2026-09"

#: LS control names, from `SPECIES_LABELING_CONFIG_XML`. Every control targets
#: the `image` object; a mismatched `to_name` makes LS drop the prediction
#: silently rather than erroring.
_IMAGE_OBJECT = "image"
_SPECIES_CONTROL = "species"
_TAXONOMY_CONTROLS = (
    ("measurable", "fish_measurable_category"),
    ("fishAngles", "fish_angle_category"),
    ("fishCurve", "fish_curved_category"),
)


class _HasJudgement(Protocol):
    """The four attributes this module reads. `SpeciesLabel` satisfies it."""

    content_of_image: Optional[str]
    fish_measurable_category: Optional[str]
    fish_angle_category: Optional[str]
    fish_curved_category: Optional[str]


def _taxonomy_result(from_name: str, path: list[str]) -> Dict[str, Any]:
    return {
        "from_name": from_name,
        "to_name": _IMAGE_OBJECT,
        "type": "taxonomy",
        "value": {"taxonomy": [path]},
    }


def build_prediction(label: _HasJudgement) -> Optional[Dict[str, Any]]:
    """Render `label` as an LS prediction, or None if it says nothing.

    `content_of_image` is a ", "-joined taxonomy path, so it splits back into
    the nested path LS wants. The three attribute taxonomies are flat, and the
    parser reads them with `taxonomy[0][0]` — the first element of the first
    path — so each must be emitted as a **single-element** path or the parser
    would read a parent where a leaf was meant.
    """
    results: list[Dict[str, Any]] = []

    content = getattr(label, "content_of_image", None)
    if content:
        results.append(_taxonomy_result(_SPECIES_CONTROL, content.split(", ")))

    for control, attribute in _TAXONOMY_CONTROLS:
        value = getattr(label, attribute, None)
        if value:
            results.append(_taxonomy_result(control, [value]))

    if not results:
        return None
    return {"model_version": PREANNOTATION_MODEL_VERSION, "result": results}
