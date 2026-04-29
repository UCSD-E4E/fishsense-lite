"""SDK ↔ API model drift guard.

The fishsense-api-sdk hand-mirrors fishsense-api's SQLModel definitions. Any
field-level drift between the two breaks the SDK at runtime — typically as a
Pydantic ValidationError when an API response carries an unexpected shape, or
as a silent acceptance on the SDK side of a payload the API would reject.

This test parametrizes every shared model and asserts that the field set and
per-field annotation are structurally equivalent. Annotations are compared by
qualified name so that types redeclared on both sides (e.g. `Priority`) are
treated as equivalent.

Some SDK models intentionally split into a private wire-format class plus an
ergonomic public class (e.g. `_CameraIntrinsics` / `CameraIntrinsics`); the
parametrization below points the comparison at the wire-format class.
"""

from __future__ import annotations

import importlib
import types
import typing
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class Pair:
    sdk_module: str
    sdk_class: str
    api_module: str
    api_class: str

    @property
    def label(self) -> str:
        return self.api_class


# Common case: same module + class name in both packages. Diverges where the
# SDK keeps a public ergonomic wrapper around a private wire-format class, or
# where the API splits a single SDK shape into separate persistence + JSON
# classes.
MODEL_PAIRS: list[Pair] = [
    Pair("camera", "Camera", "camera", "Camera"),
    Pair("camera_intrinsics", "_CameraIntrinsics", "camera_intrinsics", "CameraIntrinsics"),
    Pair("dive", "Dive", "dive", "Dive"),
    Pair("dive_frame_cluster", "DiveFrameCluster", "dive_frame_cluster", "DiveFrameClusterJson"),
    Pair("dive_slate", "DiveSlate", "dive_slate", "DiveSlate"),
    Pair("dive_slate_label", "DiveSlateLabel", "dive_slate_label", "DiveSlateLabel"),
    Pair("fish", "Fish", "fish", "Fish"),
    Pair("headtail_label", "HeadTailLabel", "head_tail_label", "HeadTailLabel"),
    Pair("image", "Image", "image", "Image"),
    Pair("laser_extrinsics", "_LaserExtrinsics", "laser_extrinsics", "LaserExtrinsics"),
    Pair("laser_label", "LaserLabel", "laser_label", "LaserLabel"),
    Pair("measurement", "Measurement", "measurement", "Measurement"),
    Pair("species", "Species", "species", "Species"),
    Pair("species_label", "SpeciesLabel", "species_label", "SpeciesLabel"),
    Pair("user", "User", "user", "User"),
]


# Known annotation drifts we accept for now, keyed by (api_class, field_name).
# The test reports any *new* drift (anything not in this set) as a failure,
# so future divergences get caught on PR. Each entry must carry a reason so
# the team knows why we're tolerating it. Drop entries from this set as the
# underlying drift is reconciled.
KNOWN_FIELD_DRIFT: dict[tuple[str, str], str] = {
    # Across the 4 label tables, the SDK accepts a JSON-encoded *string* in
    # addition to a dict; the API SQLModel only accepts a dict (the column is
    # postgres JSON). Fixing direction is a product call (does the SDK still
    # need to round-trip a stringified payload?). Tracked in
    # project_sdk_model_mirror memory.
    ("DiveSlateLabel", "label_studio_json"): "SDK accepts str; API does not.",
    ("HeadTailLabel", "label_studio_json"): "SDK accepts str; API does not.",
    ("LaserLabel", "label_studio_json"): "SDK accepts str; API does not.",
    ("SpeciesLabel", "label_studio_json"): "SDK accepts str; API does not.",
}


def _normalize(annotation: object) -> object:
    """Hashable, module-independent representation of a type annotation.

    Two annotations referencing the same *named* class/enum compare equal even
    if they originate from different modules — e.g. `Priority` is redeclared
    on both sides.
    """
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is types.UnionType or origin is typing.Union:
        return ("Union", frozenset(_normalize(a) for a in args))
    if not args:
        if annotation is type(None):  # pylint: disable=unidiomatic-typecheck
            return "None"
        return getattr(annotation, "__qualname__", repr(annotation))
    head = getattr(origin, "__qualname__", repr(origin))
    return (head, tuple(_normalize(a) for a in args))


def _load(pair: Pair) -> tuple[type, type]:
    sdk = getattr(
        importlib.import_module(f"fishsense_api_sdk.models.{pair.sdk_module}"),
        pair.sdk_class,
    )
    api = getattr(
        importlib.import_module(f"fishsense_api.models.{pair.api_module}"),
        pair.api_class,
    )
    return sdk, api


@pytest.mark.parametrize("pair", MODEL_PAIRS, ids=lambda p: p.label)
def test_field_names_match(pair: Pair) -> None:
    sdk, api = _load(pair)
    only_sdk = set(sdk.model_fields) - set(api.model_fields)
    only_api = set(api.model_fields) - set(sdk.model_fields)
    assert not only_sdk and not only_api, (
        f"{pair.label} field set drift — "
        f"SDK only: {sorted(only_sdk)}, API only: {sorted(only_api)}"
    )


@pytest.mark.parametrize("pair", MODEL_PAIRS, ids=lambda p: p.label)
def test_field_annotations_match(pair: Pair) -> None:
    sdk, api = _load(pair)
    mismatches = []
    for fname in set(sdk.model_fields) & set(api.model_fields):
        if (pair.label, fname) in KNOWN_FIELD_DRIFT:
            continue
        sdk_t = _normalize(sdk.model_fields[fname].annotation)
        api_t = _normalize(api.model_fields[fname].annotation)
        if sdk_t != api_t:
            mismatches.append(
                f"  {fname}: SDK={sdk.model_fields[fname].annotation!r}"
                f"  vs  API={api.model_fields[fname].annotation!r}"
            )
    assert not mismatches, f"{pair.label} annotation drift:\n" + "\n".join(mismatches)


def test_known_drift_entries_are_still_drifting() -> None:
    """Allowlist hygiene: an entry in KNOWN_FIELD_DRIFT must correspond to
    real, live drift. If a field has been reconciled, drop the allowlist
    entry — leaving stale entries silently masks future regressions."""
    pair_by_label = {p.label: p for p in MODEL_PAIRS}
    stale: list[str] = []
    for (label, fname), _reason in KNOWN_FIELD_DRIFT.items():
        pair = pair_by_label.get(label)
        if pair is None:
            stale.append(f"{label}.{fname}: no MODEL_PAIRS entry for {label!r}")
            continue
        sdk, api = _load(pair)
        sdk_field = sdk.model_fields.get(fname)
        api_field = api.model_fields.get(fname)
        if sdk_field is None or api_field is None:
            stale.append(f"{label}.{fname}: field no longer exists on both sides")
            continue
        if _normalize(sdk_field.annotation) == _normalize(api_field.annotation):
            stale.append(
                f"{label}.{fname}: SDK and API now match — drop this allowlist entry"
            )
    assert not stale, "Stale KNOWN_FIELD_DRIFT entries:\n  " + "\n  ".join(stale)
