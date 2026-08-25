"""The `SpeciesLabel.content_of_image` taxonomy vocabulary.

Label Studio writes the labeler's taxonomy selection into
`SpeciesLabel.content_of_image` as a ", "-joined path:

    "Fish, Hogfish (Lachnolaimus maximus)"  -> a real (wild) fish
    "Fish Model, Weasly Fish"               -> a rigid model
    "Calibration Targets, Ruler"            -> the ruler
    "Slate, Laser on slate"                 -> a slate frame (stage 9)

Four consumers read that string, and they used to spell the markers
independently: `measure_fish_activity` (Python, data-worker), the stage-14
cohort selector (SQLAlchemy, api), the `dive_pipeline_status` view (raw SQL,
api), and the stage-9 slate cohort. Keeping them in step was a comment-only
contract — and it had already failed. Both the controller and the view
carried a worked example claiming `"Fish Model, …"` and
`"Calibration Targets, Ruler"` were *skipped*, six lines above code that
matched both as measurable; a maintainer trusting it would have deleted the
ruler clause and silently broken ruler validation.

Now the literals live here once, and every consumer derives from them.

**Python vs SQL.** `is_measurable` is the definition of record: measurable
means `measure_fish_activity` will actually bind a Measurement. The SQL and
SQLAlchemy predicates are `LIKE`-based *approximations* of it, because the
view has to run on Postgres in prod and SQLite under test, which rules out
the string functions an exact port would need. `MEASURABILITY_CORPUS` is the
shared fixture that keeps the approximation honest:
`test_dive_pipeline_status_view.py` runs the real SQL over it and asserts
agreement with `is_measurable`. The known ceiling is that `REAL_FISH_LIKE`
tests the whole string for "contains `(` … ends with `)`" while
`parse_species_names` tests only the final chunk, so a value whose only
parens sit in an *earlier* chunk and which still ends in `)` would diverge
("Fish (a), Hogfish)"). No taxonomy branch produces that shape; if one ever
does, add it to the corpus and the parity test will fail rather than let the
cohort and the activity disagree.
"""

from __future__ import annotations

# --- Literals, exactly as the Label Studio labeling config emits them ------

# Prefix for a physical fish model. The leaf after it is the model's identity
# — the `name` natural key on Fish.
FISH_MODEL_PREFIX = "Fish Model,"

# The ruler is a rigid known-length target like the models, so it measures
# through the same name-keyed path. Unlike a fish model its endpoints are
# unambiguous — no tip-vs-fork landmark uncertainty — so it isolates
# calibration error from labeling convention.
# Every `Fish Model, <name>` leaf a labeler can pick, in species-XML order.
#
# Lives here rather than in either service because the two halves are split
# across packages: the labeling XML that offers these choices is in the
# api-worker, and the `fishmodelreference` rows that make a measurement
# *gradeable* are in the API. Nothing connected them, and the gap is silent —
# `fish_model_measurement_accuracy` inner-joins on `Fish.name`, so a model with
# no reference row produces no rows at all. No error, no NULL, just absence.
#
# `Weasly Fish` sat in exactly that state in prod: pickable, measurable, and
# graded by nobody.
#
# Kept in step from both sides: the api-worker asserts the XML matches this
# list, and the API asserts every name here has a reference row. Adding a model
# means editing the XML, this list, and `views.KNOWN_FISH_MODELS` — the tests
# name whichever you forget.
LABELED_FISH_MODELS = (
    "Weasly Fish",
    "Snook",
    "Grouper",
    "Shark",
    "Gray Anthias",
    "Purple Angel",
    "Yellow Anthias",
)

RULER_CONTENT = "Calibration Targets, Ruler"
RULER_NAME = "Ruler"

# Stage-9 marker: the frame shows the slate with the laser on it. This is read
# off `taxonomy[0]`, a separate path from the slate-*type* leaf that species
# sync maps to `Dive.dive_slate_id`.
SLATE_CONTENT_MARKER = "Slate, Laser on slate"

# --- SQL LIKE patterns, for the SQLAlchemy + raw-SQL consumers -------------

# A real fish carries a `Common Name (Scientific name)` leaf. `(` and `)` are
# not LIKE wildcards, so this reads "contains ( and ends with )".
REAL_FISH_LIKE = "%(%)"
FISH_MODEL_LIKE = f"{FISH_MODEL_PREFIX}%"

__all__ = [
    "FISH_MODEL_LIKE",
    "LABELED_FISH_MODELS",
    "FISH_MODEL_PREFIX",
    "MEASURABILITY_CORPUS",
    "REAL_FISH_LIKE",
    "RULER_CONTENT",
    "RULER_NAME",
    "SLATE_CONTENT_MARKER",
    "SQL_BROADER_THAN_PYTHON",
    "is_measurable",
    "measurable_species_sql",
    "parse_model_name",
    "parse_species_names",
    "rigid_target_sql",
]


def rigid_target_sql(col: str) -> str:
    """SQL for "this row is a fish model or the ruler".

    The `TRIM(...) <> prefix` half is not decoration. `LIKE 'Fish Model,%'`
    matches the *empty leaf* `"Fish Model,"` — a labeler selecting the parent
    taxonomy node without picking a model — because `%` matches the empty
    string. `parse_model_name` returns None for it, so `measure_fish_activity`
    skips the image. Cohort says measurable, activity says skip: no
    Measurement is ever written, `NOT EXISTS (measurement)` stays true, and
    the dive is re-selected every hour forever. Exactly the never-goes-false
    wedge that blocked scheduling stage 14 before 2026-07-17, reachable by one
    labeler mis-click.

    `TRIM` also covers a space-only leaf (`"Fish Model,   "`), which
    `parse_model_name` rejects via the matching `.strip(" ")`. Note SQL
    `TRIM(x)` removes **spaces only**, not all whitespace — which is why
    `parse_model_name` strips spaces only too, rather than calling bare
    `.strip()`. Both `TRIM` and the comparison behave identically on Postgres
    (prod) and SQLite (tests).
    """
    return (
        f"(({col} LIKE '{FISH_MODEL_LIKE}' "
        f"AND TRIM({col}) <> '{FISH_MODEL_PREFIX}') "
        f"OR {col} = '{RULER_CONTENT}')"
    )


def measurable_species_sql(col: str) -> str:
    """SQL approximation of `is_measurable` — real fish OR rigid target."""
    return f"({col} LIKE '{REAL_FISH_LIKE}' OR {rigid_target_sql(col)})"


def parse_species_names(content_of_image: str | None) -> tuple[str, str] | None:
    """Pull `(common_name, scientific_name)` out of a real fish's taxonomy path.

    Format: `"..., Common Name (Scientific name)"`. Returns None if the field
    is empty or off-shape — we skip rather than write a malformed Species row.

    The shape test is a bare `"("`, deliberately, even though the extraction
    below splits on `" ("`. Tightening it to `" ("` looks like an improvement
    — it would skip `"Fish, Hogfish(Lachnolaimus maximus)"` instead of
    splitting it into nonsense — but `REAL_FISH_LIKE` cannot express
    "space before the paren" over the whole string, so the tightening made
    Python reject rows the SQL still matched. That direction is the
    never-drains wedge: cohort offers, activity skips, no Measurement, dive
    re-selected forever. Agreeing with the SQL matters more than rejecting a
    shape the Label Studio taxonomy cannot emit, so the guard stays loose and
    `MEASURABILITY_CORPUS` pins the agreement.
    """
    if not content_of_image:
        return None
    last_chunk = content_of_image.split(", ")[-1]
    if "(" not in last_chunk or not last_chunk.endswith(")"):
        return None
    common = last_chunk.split(" (")[0].strip()
    scientific = last_chunk.split(" (")[-1][:-1].strip()
    if not common or not scientific:
        return None
    return common, scientific


def parse_model_name(content_of_image: str | None) -> str | None:
    """Return the target name for a rigid known-length target, else None.

    Covers `"Fish Model, <name>"` and the ruler
    (`"Calibration Targets, Ruler"` -> `"Ruler"`). Real fish and every other
    branch return None. An empty leaf (`"Fish Model,"` with nothing after)
    returns None — nothing to identify — matching the "skip rather than write
    a malformed row" posture of `parse_species_names`.
    """
    if not content_of_image:
        return None
    if content_of_image.strip() == RULER_CONTENT:
        return RULER_NAME
    if not content_of_image.startswith(FISH_MODEL_PREFIX):
        return None
    # `.strip(" ")`, not `.strip()`: the SQL guard is `TRIM(col)`, which removes
    # spaces only. Stripping all whitespace here would make Python reject
    # "Fish Model,\t" while the SQL still matched it — the wedge again.
    name = content_of_image[len(FISH_MODEL_PREFIX) :].strip(" ")
    return name or None


def is_measurable(content_of_image: str | None) -> bool:
    """True iff `measure_fish_activity` can bind this row to a Measurement.

    This is the definition of record; the `LIKE` predicates approximate it.
    Anything looser makes the stage-14 cohort offer an image the activity
    always skips: no Measurement is written, `NOT EXISTS (measurement)` stays
    true, and the dive is re-selected every hour forever.
    """
    return (
        parse_species_names(content_of_image) is not None
        or parse_model_name(content_of_image) is not None
    )


# --- Shared parity fixture -------------------------------------------------

# `(content_of_image, is_measurable)` over every branch that actually occurs
# plus the boundary cases. Used by this package's unit tests AND by the api's
# `test_dive_pipeline_status_view.py`, which runs the real SQL predicate over
# it — that cross-check is what stops the Python and SQL representations
# drifting apart again.
MEASURABILITY_CORPUS: tuple[tuple[str | None, bool], ...] = (
    # Real fish — the `Common (Scientific)` leaf.
    ("Fish, Hogfish (Lachnolaimus maximus)", True),
    ("Fish, Stoplight Parrotfish (Sparisoma viride)", True),
    ("Fish, Bar Jack (Caranx ruber)", True),
    # Rigid known-length targets — name-keyed, no parens.
    ("Fish Model, Weasly Fish", True),
    ("Fish Model, Snook", True),
    ("Fish Model, Purple Angel", True),
    ("Calibration Targets, Ruler", True),
    # Shapes the taxonomy cannot emit, kept because they are where the SQL
    # approximation and the Python parser are most likely to drift apart. All
    # of these parse into junk names — that is accepted deliberately: the two
    # representations agreeing matters more than rejecting an unreachable
    # value, because "SQL matches, Python skips" is the never-drains wedge
    # while "both accept junk" is merely a junk row on input that never comes.
    ("Fish, Hogfish(Lachnolaimus maximus)", True),  # no space before the paren
    ("Fish, ()", True),
    ("Fish Model,\t", True),  # tab leaf: SQL TRIM removes spaces only
    # Not measurable.
    ("Slate, Laser on slate", False),
    ("Calibration Targets, Slate", False),
    ("Fish Model,", False),  # empty leaf — parent node, no model picked
    ("Fish Model,   ", False),  # whitespace-only leaf
    ("", False),
    (None, False),
)


# Shapes where the SQL approximation is BROADER than `is_measurable`, i.e. the
# view/cohort would call them measurable and `measure_fish_activity` would skip
# them. That direction is the dangerous one — it is the never-goes-false wedge
# — so the set is pinned rather than left to chance:
# `test_dive_pipeline_status_view.py` asserts the SQL matches exactly these and
# nothing else, so a *new* divergence fails the build instead of reaching prod.
#
# The whole class comes from one thing the `LIKE` patterns cannot express: the
# empty-name guard in `parse_species_names`. `%(%)` sees "contains ( and ends
# with )" and has no way to check that the pieces either side of the paren are
# non-empty. Closing it would need Postgres-only string functions, which the
# view can't use because the tests run it on SQLite.
#
# None of these are producible by the Label Studio taxonomy config — every leaf
# is a fixed choice with a non-empty common and scientific name. If one ever
# becomes reachable, fix the guard rather than extending this tuple.
SQL_BROADER_THAN_PYTHON: tuple[str, ...] = (
    "Fish,  (Lachnolaimus maximus)",  # empty common name
    "Fish, Hogfish ()",  # empty scientific name
)
