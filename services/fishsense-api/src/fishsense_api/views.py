"""SQL view definitions exposed alongside the SQLModel tables.

Views are not part of `SQLModel.metadata`; they are created out-of-band
by alembic in prod and by the test fixtures in CI. This module owns
the canonical SQL string so both code paths apply the same definition.

The shape lives here (not in `models/`) because views are derived
artifacts, not entities — there's no row-level CRUD against them.
Superset reads them directly via the `fishsense` Postgres connection.

Portability note: the SQL is plain ANSI/SQL-92 with `EXISTS` / `NOT
EXISTS` subqueries. Both the prod Postgres and the in-memory sqlite
the test fixture uses parse it identically. No `bool_and` /
`COUNT(...) FILTER` (Postgres-only) — adding those would force the
test suite to spin up a real Postgres.
"""

from __future__ import annotations

from fishsense_shared import taxonomy

DIVE_PIPELINE_STATUS_VIEW_NAME = "dive_pipeline_status"

# Stage-9 slate marker. Sourced from `fishsense_shared.taxonomy` rather than
# from `dive_cohort_controller`, which keeps the old constraint satisfied —
# `views.py` must not pull controller imports, because alembic's env loads
# this module during migrations, before the FastAPI app exists — while still
# giving the view, the cohort selector and the data-worker one definition.
_SLATE_CONTENT_MARKER = taxonomy.SLATE_CONTENT_MARKER

# A *valid* laser label: the labeler placed a point, the validator signed
# off, and `ValidateLaserLabelsForDiveWorkflow`'s RANSAC fit hasn't
# superseded it. This is the gate stages 1, 2, 5.1 and 14 all cascade
# from, so it's spelled once here rather than five times inline. Assumes
# the enclosing subquery aliases laserlabel as `ll`.
_VALID_LASER_SQL = """ll.completed = TRUE
           AND ll.superseded = FALSE
           AND ll.x IS NOT NULL
           AND ll.y IS NOT NULL"""

# A *valid* head/tail label: both keypoints fully placed. Only stage 14
# needs this — every other stage only cares that a HeadTailLabel row
# exists — but it mirrors `_VALID_LASER_SQL` so the pair reads together.
# Assumes the enclosing subquery aliases headtaillabel as `htl`.
# The `LaserExtrinsics.id` a dive is actually processed with: its own row,
# else the one borrowed through `calibration_dive_id`. Own-then-link is the
# order `get_laser_extrinsics_for_dive` applies, and `uq_laserextrinsics_dive_id`
# guarantees at most one row per dive, so neither branch needs a tie-break.
# Assumes the enclosing query aliases dive as `d`.
_RESOLVED_EXTRINSICS_SQL = """COALESCE(
                     (SELECT le.id FROM laserextrinsics le WHERE le.dive_id = d.id),
                     (SELECT le.id FROM laserextrinsics le
                      WHERE le.dive_id = d.calibration_dive_id)
                 )"""


_VALID_HEADTAIL_SQL = """htl.completed = TRUE
                 AND htl.superseded = FALSE
                 AND htl.head_x IS NOT NULL
                 AND htl.head_y IS NOT NULL
                 AND htl.tail_x IS NOT NULL
                 AND htl.tail_y IS NOT NULL"""

# An image stage 14 can actually turn into a Measurement.
#
# Three measurable branches, matching `taxonomy.is_measurable`:
#
#     "Fish, Hogfish (Lachnolaimus maximus)"  -> real fish (parens leaf)
#     "Fish Model, Weasly Fish"               -> rigid model, name-keyed
#     "Calibration Targets, Ruler"            -> the ruler, name-keyed
#
# Everything else — "Slate, Laser on slate", other Calibration Targets — is
# not measurable. (An earlier version of this comment listed the bottom two
# as *skipped* and asserted "Calibration Targets stays unmeasurable", while
# the SQL below matched both. It predated fish models and the ruler and was
# never updated.)
#
# Without this condition the cohort and the activity disagree, and that
# disagreement cannot resolve: the selector keeps offering an image the
# activity always skips, no Measurement is ever written, so `NOT EXISTS
# (measurement)` stays true and the dive is re-selected every hour forever.
# That is the same never-goes-false shape that blocked scheduling stage 14
# before 2026-07-17.
#
# These LIKE patterns approximate `fishsense_shared.taxonomy.is_measurable`,
# which is the definition of record (it is what `measure_fish_activity`
# actually binds on). `test_dive_pipeline_status_view.py` runs this SQL over
# `taxonomy.MEASURABILITY_CORPUS` and asserts the two agree, so the Python
# and SQL representations cannot drift apart silently again.
#
# Assumes the enclosing subquery aliases specieslabel as `sl`.
_MEASURABLE_SPECIES_SQL = taxonomy.measurable_species_sql("sl.content_of_image")

# A rigid known-length target — fish model or ruler. No cluster required:
# these carry no grouping labels. Mirrors
# `dive_cohort_controller._is_fish_model_condition`.
# Assumes the enclosing subquery aliases specieslabel as `sl`.
_IS_FISH_MODEL_SQL = taxonomy.rigid_target_sql("sl.content_of_image")

# "Complete" everywhere = ≥1 completed-non-superseded row AND zero
# incomplete-non-superseded rows. Mirrors
# `get_dives_with_complete_laser_labeling`'s semantics so a dive with
# zero labels of a kind doesn't vacuously read as complete.
# Every image correlation below reads `i.dive_id = d.id AND i.is_canonical`.
#
# The same physical frames live under several dive rows (half of prod's image
# table is duplicate content; 207 of 479 dives are duplicates end to end), and
# `is_canonical` marks which copy is real. The cohort selectors in
# `dive_cohort_controller` gate on it so a duplicate dive can never become
# pipeline work, and this view has to use the identical predicate or the two drift:
# the dashboard would report a dive as perpetually incomplete while the worker
# correctly does nothing with it. `test_dive_pipeline_status_view.py` pins that
# agreement explicitly.

DIVE_PIPELINE_STATUS_VIEW_SQL = f"""
CREATE VIEW {DIVE_PIPELINE_STATUS_VIEW_NAME} AS
SELECT
    d.id AS dive_id,
    d.name AS dive_name,
    d.priority,
    d.dive_slate_id,

    -- Stage 0.1: every image in the dive has at least one LaserLabel
    -- row (any project, any state). The cohort selector is the
    -- inverse of this predicate.
    (EXISTS (SELECT 1 FROM image i WHERE i.dive_id = d.id AND i.is_canonical)
     AND NOT EXISTS (
         SELECT 1 FROM image i
         WHERE i.dive_id = d.id AND i.is_canonical
           AND NOT EXISTS (
               SELECT 1 FROM laserlabel ll WHERE ll.image_id = i.id
           )
     )) AS laser_preprocessed,

    -- Laser labeling: ≥1 completed-non-superseded AND zero
    -- incomplete-non-superseded.
    (EXISTS (
         SELECT 1 FROM laserlabel ll
         JOIN image i ON i.id = ll.image_id
         WHERE i.dive_id = d.id AND i.is_canonical
           AND ll.superseded = FALSE
           AND ll.completed = TRUE
     )
     AND NOT EXISTS (
         SELECT 1 FROM laserlabel ll
         JOIN image i ON i.id = ll.image_id
         WHERE i.dive_id = d.id AND i.is_canonical
           AND ll.superseded = FALSE
           AND (ll.completed = FALSE OR ll.completed IS NULL)
     )) AS laser_labeling_complete,

    -- Stage 5.1: every image carrying a *valid* laser label
    -- (completed, not superseded, x/y both set) has a non-sentinel
    -- HeadTailLabel row. ≥1 such image must exist (otherwise the
    -- predicate would be vacuously true for dives with no laser work
    -- yet).
    (EXISTS (
         SELECT 1 FROM laserlabel ll
         JOIN image i ON i.id = ll.image_id
         WHERE i.dive_id = d.id AND i.is_canonical
           AND {_VALID_LASER_SQL}
     )
     AND NOT EXISTS (
         SELECT 1 FROM laserlabel ll
         JOIN image i ON i.id = ll.image_id
         WHERE i.dive_id = d.id AND i.is_canonical
           AND {_VALID_LASER_SQL}
           AND NOT EXISTS (
               SELECT 1 FROM headtaillabel htl
               WHERE htl.image_id = i.id
                 AND htl.label_studio_project_id IS NOT NULL
           )
     )) AS headtail_preprocessed,

    -- Headtail labeling: ≥1 completed-non-superseded AND zero
    -- incomplete-non-superseded.
    (EXISTS (
         SELECT 1 FROM headtaillabel htl
         JOIN image i ON i.id = htl.image_id
         WHERE i.dive_id = d.id AND i.is_canonical
           AND htl.superseded = FALSE
           AND htl.completed = TRUE
     )
     AND NOT EXISTS (
         SELECT 1 FROM headtaillabel htl
         JOIN image i ON i.id = htl.image_id
         WHERE i.dive_id = d.id AND i.is_canonical
           AND htl.superseded = FALSE
           AND (htl.completed = FALSE OR htl.completed IS NULL)
     )) AS headtail_labeling_complete,

    -- Stage 1: ≥1 PREDICTION cluster present (stage-1 clustering ran
    -- and persisted output). The data-worker stage 1 workflow does not
    -- yet write back, so this is currently false dive-wide; the column
    -- still exists so dashboards don't have to rebuild when stage 1
    -- starts persisting.
    EXISTS (
        SELECT 1 FROM diveframecluster dfc
        WHERE dfc.dive_id = d.id
          AND dfc.data_source = 'PREDICTION'
    ) AS has_prediction_clusters,

    -- Stage 2: every "processable" image has a non-sentinel, non-superseded
    -- SpeciesLabel row, and ≥1 processable image exists (else vacuously true
    -- for dives with no laser work yet — same convention as
    -- headtail_preprocessed).
    --
    -- "Processable" = carries a valid laser label (completed, not superseded,
    -- x/y both set) AND is itself in a PREDICTION cluster. Both halves mirror
    -- `select_next_for_species_preprocessing` exactly, and the drift-guard
    -- test pins the two together:
    --   * in-cluster (2026-07-22): the resolver only makes JPEGs for a
    --     qualifying image that is in a cluster (it needs the cluster for the
    --     "image i of N" overlay). An image with a valid laser but no cluster
    --     — a laser validated after one-shot stage-1 clustering — is not
    --     processable, so it must not count against "preprocessed" (it would
    --     otherwise read as forever-incomplete while the selector, correctly,
    --     never re-fired on it).
    --   * superseded (2026-07-22): a dead-lettered species row is not
    --     evidence the work is done — matches the selector's superseded gate.
    (EXISTS (
         SELECT 1 FROM laserlabel ll
         JOIN image i ON i.id = ll.image_id
         WHERE i.dive_id = d.id AND i.is_canonical
           AND {_VALID_LASER_SQL}
           AND EXISTS (
               SELECT 1 FROM diveframeclusterimagemapping mm
               JOIN diveframecluster dfc
                 ON dfc.id = mm.dive_frame_cluster_id
               WHERE mm.image_id = i.id
                 AND dfc.data_source = 'PREDICTION'
           )
     )
     AND NOT EXISTS (
         SELECT 1 FROM laserlabel ll
         JOIN image i ON i.id = ll.image_id
         WHERE i.dive_id = d.id AND i.is_canonical
           AND {_VALID_LASER_SQL}
           AND EXISTS (
               SELECT 1 FROM diveframeclusterimagemapping mm
               JOIN diveframecluster dfc
                 ON dfc.id = mm.dive_frame_cluster_id
               WHERE mm.image_id = i.id
                 AND dfc.data_source = 'PREDICTION'
           )
           AND NOT EXISTS (
               SELECT 1 FROM specieslabel sl
               WHERE sl.image_id = i.id
                 AND sl.label_studio_project_id IS NOT NULL
                 AND sl.superseded = FALSE
           )
     )) AS dive_images_preprocessed,

    -- Species labeling: ≥1 completed-non-superseded AND zero
    -- incomplete-non-superseded (mirrors laser/headtail — a superseded
    -- row is dead-lettered, so it neither satisfies nor blocks completion).
    (EXISTS (
         SELECT 1 FROM specieslabel sl
         JOIN image i ON i.id = sl.image_id
         WHERE i.dive_id = d.id AND i.is_canonical
           AND sl.superseded = FALSE
           AND sl.completed = TRUE
     )
     AND NOT EXISTS (
         SELECT 1 FROM specieslabel sl
         JOIN image i ON i.id = sl.image_id
         WHERE i.dive_id = d.id AND i.is_canonical
           AND sl.superseded = FALSE
           AND (sl.completed = FALSE OR sl.completed IS NULL)
     )) AS species_labeling_complete,

    -- Slate path applies only when dive has an associated slate.
    (d.dive_slate_id IS NOT NULL) AS slate_applicable,

    -- Stage 9: every image whose SpeciesLabel marks it as containing a
    -- slate-with-laser has a non-sentinel DiveSlateLabel row. ≥1 such
    -- image must exist.
    (d.dive_slate_id IS NOT NULL
     AND EXISTS (
         SELECT 1 FROM specieslabel sl
         JOIN image i ON i.id = sl.image_id
         WHERE i.dive_id = d.id AND i.is_canonical
           AND sl.content_of_image = '{_SLATE_CONTENT_MARKER}'
     )
     AND NOT EXISTS (
         SELECT 1 FROM specieslabel sl
         JOIN image i ON i.id = sl.image_id
         WHERE i.dive_id = d.id AND i.is_canonical
           AND sl.content_of_image = '{_SLATE_CONTENT_MARKER}'
           AND NOT EXISTS (
               SELECT 1 FROM diveslatelabel dsl
               WHERE dsl.image_id = i.id
                 AND dsl.label_studio_project_id IS NOT NULL
           )
     )) AS slate_preprocessed,

    -- Slate labeling: ≥1 completed-non-superseded AND zero
    -- incomplete-non-superseded (mirrors laser/headtail/species).
    (EXISTS (
         SELECT 1 FROM diveslatelabel dsl
         JOIN image i ON i.id = dsl.image_id
         WHERE i.dive_id = d.id AND i.is_canonical
           AND dsl.superseded = FALSE
           AND dsl.completed = TRUE
     )
     AND NOT EXISTS (
         SELECT 1 FROM diveslatelabel dsl
         JOIN image i ON i.id = dsl.image_id
         WHERE i.dive_id = d.id AND i.is_canonical
           AND dsl.superseded = FALSE
           AND (dsl.completed = FALSE OR dsl.completed IS NULL)
     )) AS slate_labeling_complete,

    -- Stage 13: calibration is available for the dive — either its own
    -- LaserExtrinsics row, or a borrowed one via `calibration_dive_id`
    -- (a fish-only dive pointing at a sibling slate dive shot with the
    -- same rig). Mirrors `get_laser_extrinsics_for_dive`'s resolution and
    -- the stage-14 cohort's `has_laser_extrinsics`.
    (EXISTS (
        SELECT 1 FROM laserextrinsics le
        WHERE le.dive_id = d.id
    ) OR EXISTS (
        SELECT 1 FROM laserextrinsics le
        WHERE le.dive_id = d.calibration_dive_id
    )) AS calibrated,

    -- WHERE the calibration came from — the strongest data-quality signal on a
    -- measurement. `calibrated` only says a dive HAS extrinsics; provenance
    -- says whether they describe THIS dive's rig deployment.
    --
    -- Measured 2026-08-04 against the known-length fish models: a dive on its
    -- own slate measures to ~1%, while a BORROWED calibration carries an extra
    -- rig-state systematic of -8..+2% (n=7, all seven model dives borrow).
    -- That error is irreducible in software — laser-dot precision (~0.5%) and
    -- the line fit (already at the label-noise floor) were both ruled out by
    -- experiment — so downstream analysis must be able to filter or weight on
    -- this rather than treat every measurement as equally trustworthy.
    --
    -- 'own' wins over 'borrowed' to mirror `get_laser_extrinsics_for_dive`'s
    -- own-then-link resolution: a dive with its own row never uses the link.
    CASE
        WHEN EXISTS (
            SELECT 1 FROM laserextrinsics le WHERE le.dive_id = d.id
        ) THEN 'own'
        WHEN EXISTS (
            SELECT 1 FROM laserextrinsics le
            WHERE le.dive_id = d.calibration_dive_id
        ) THEN 'borrowed'
        ELSE 'none'
    END AS calibration_source,

    -- Stage 14: ≥1 measurement for the dive AND no measurable image left
    -- unmeasured *under the current calibration*. "Measurable" mirrors what
    -- measure_fish_activity actually attempts: a top-three species label
    -- whose image carries a valid laser label, a valid headtail label, and a
    -- LABEL_STUDIO cluster.
    --
    -- The calibration qualifier is not decoration. A length is a function of
    -- the extrinsics behind its depth, so replacing a dive's calibration
    -- invalidates every length computed from the old one — and that happens:
    -- the 2026-08-11 slate panel-offset fix recalibrated 6 of the 8 dives
    -- that already had measurements, silently leaving their lengths wrong
    -- while this column still read true. An image counts as measured only
    -- when its Measurement names the extrinsics row that would be resolved
    -- today. Rows written before `measurement.laser_extrinsics_id` existed
    -- carry NULL and so read as unmeasured until stage 14 revisits them,
    -- which is the intended (and self-draining) backfill.
    --
    -- Rescoped 2026-07-17. The predicate used to be "≥1 LABEL_STUDIO
    -- cluster AND zero with fish_id NULL", which was unreachable: a
    -- cluster is only bound to a fish through a measurable image, so any
    -- cluster without one held the dive at measured=false forever. Prod
    -- had all 8 calibrated dives pinned that way (dive 466: 1632 unbound
    -- clusters vs 24 measurable images, the residue of repeated stage-6.1
    -- POSTs). Because the stage-14 cohort mirrors this predicate, a
    -- scheduled stage 14 would also have re-selected the same dives
    -- forever.
    (EXISTS (
         SELECT 1 FROM measurement m
         JOIN image i ON i.id = m.image_id
         WHERE i.dive_id = d.id AND i.is_canonical
     )
     AND NOT EXISTS (
         SELECT 1 FROM specieslabel sl
         JOIN image i ON i.id = sl.image_id
         WHERE i.dive_id = d.id AND i.is_canonical
           AND sl.top_three_photos_of_group = TRUE
           AND {_MEASURABLE_SPECIES_SQL}
           AND EXISTS (
               SELECT 1 FROM laserlabel ll
               WHERE ll.image_id = i.id
                 AND {_VALID_LASER_SQL}
           )
           AND EXISTS (
               SELECT 1 FROM headtaillabel htl
               WHERE htl.image_id = i.id
                 AND {_VALID_HEADTAIL_SQL}
           )
           AND (
               EXISTS (
                   SELECT 1 FROM diveframeclusterimagemapping mm
                   JOIN diveframecluster dfc
                     ON dfc.id = mm.dive_frame_cluster_id
                   WHERE mm.image_id = i.id
                     AND dfc.data_source = 'LABEL_STUDIO'
               )
               OR {_IS_FISH_MODEL_SQL}
           )
           AND NOT EXISTS (
               SELECT 1 FROM measurement m
               WHERE m.image_id = i.id
                 AND m.laser_extrinsics_id = {_RESOLVED_EXTRINSICS_SQL}
           )
     )) AS measured

FROM dive d
"""

DROP_DIVE_PIPELINE_STATUS_VIEW_SQL = (
    f"DROP VIEW IF EXISTS {DIVE_PIPELINE_STATUS_VIEW_NAME}"
)


# ---------------------------------------------------------------------------
# Fish-model measurement accuracy
# ---------------------------------------------------------------------------

FISH_MODEL_ACCURACY_VIEW_NAME = "fish_model_measurement_accuracy"

# Ground-truth lengths of the physical fish models, in meters, as measured by
# the team. Seeded into `fishmodelreference` by alembic; the constant is the
# source of truth so a change is reviewable in a diff.
#
# These are the pipeline's held-out VALIDATION set — together with the ruler
# (`Calibration Targets, Ruler`). They are compared against what stage 14
# produces and are NEVER fed into laser calibration: a benchmark that
# calibrates from its own answer key measures nothing. Calibration comes from
# slates only.
#
# Field model dates: 2024-08-21 and 2024-10-16.
KNOWN_FISH_MODELS = [
    {"name": "Snook", "known_length_m": 0.455},
    {"name": "Grouper", "known_length_m": 0.360},
    {"name": "Shark", "known_length_m": 0.605},
    {"name": "Gray Anthias", "known_length_m": 0.195},
    {"name": "Purple Angel", "known_length_m": 0.192},
    {"name": "Yellow Anthias", "known_length_m": 0.200},
    # The ruler is a rigid known-length target measured through the same
    # name-keyed path as the models. 13.5 in, NOT the ruler's nominal 14 in:
    # labelers click the first *printed* graduation (the 0.5 mark) and the 14
    # mark, so the labeled span is 0.5->14. Measured off the ruler's own inch
    # ticks on 4 frames: 13.500/13.505/13.481/13.468 in (SD 0.13%), and a
    # second method (half-inch ticks + 1D projective fit, which also removes
    # perspective) independently agreed. Do NOT "correct" this back to 14 in
    # without re-instructing labelers to click a physical 0 that the scale does
    # not print.
    {"name": "Ruler", "known_length_m": 0.3429},
]

# One row per fish-model measurement, with its error against the known length.
#
# Joins through `Fish.name` — the natural key stage 14 resolves model identity
# by — so every new measurement lands in the view automatically. Real (wild)
# fish carry `name IS NULL` and are excluded by the inner join; so are models
# that have no reference row yet (ungradeable rather than compared to NULL).
#
# `pct_error` is signed: positive = measured long. Length is proportional to
# the laser-derived depth, so a systematic offset here is a calibration
# signal, not a labeling one.
FISH_MODEL_ACCURACY_VIEW_SQL = f"""
CREATE VIEW {FISH_MODEL_ACCURACY_VIEW_NAME} AS
SELECT
    m.id            AS measurement_id,
    m.image_id      AS image_id,
    i.dive_id       AS dive_id,
    f.id            AS fish_id,
    f.name          AS model_name,
    r.known_length_m AS known_length_m,
    m.length_m      AS length_m,
    (m.length_m - r.known_length_m) AS error_m,
    (100.0 * (m.length_m - r.known_length_m) / r.known_length_m) AS pct_error
FROM measurement m
JOIN image i ON i.id = m.image_id
JOIN fish f ON f.id = m.fish_id
JOIN fishmodelreference r ON r.name = f.name
WHERE m.length_m IS NOT NULL
"""

DROP_FISH_MODEL_ACCURACY_VIEW_SQL = (
    f"DROP VIEW IF EXISTS {FISH_MODEL_ACCURACY_VIEW_NAME}"
)


# ---------------------------------------------------------------------------
# Per-fish length estimate
# ---------------------------------------------------------------------------

FISH_LENGTH_ESTIMATE_VIEW_NAME = "fish_length_estimate"

# One row per (fish, dive) giving the length estimate to actually USE, plus the
# alternatives for comparison.
#
# **Use `length_p90_m`, not the mean.** Stage 14 back-projects head and tail at
# a single laser-derived depth, so it measures the fish's *projection*: any
# out-of-plane angle can only ever SHORTEN it. Per-frame error is therefore
# one-sided-negative — measured on 437 fish-model frames, the within-group IQR
# is only 2.2pp and the median +0.3%, but the skew is -4.87, with 2.3% of
# frames below -10pp against 0.5% above +10pp. Averaging drags the estimate
# into that tail; a high quantile rejects it.
#
# Measured over 23 dive x model groups (n>=8): mean 4.35% absolute error,
# median 3.58%, p75 2.68%, p90 2.26%. p90 rather than max because max chases
# the single most-favourable frame and so inherits its label noise, while p90
# keeps the one-sided-tail rejection.
#
# Covers every fish, not just models: `species_id` and `name` are both exposed
# so wild fish (name NULL) and models (species_id NULL) are distinguishable.
# `n_frames` is the honest caveat — a p90 over 2 frames is not a p90, so
# consumers should filter on it.
# Nearest-rank quantiles via ROW_NUMBER rather than `percentile_cont`, because
# the view tests run on SQLite and it has no ordered-set aggregates. Integer
# arithmetic gives ceil(q*n) portably: (9n+9)/10 = ceil(0.9n), (n+1)/2 =
# ceil(0.5n). Nearest-rank also avoids interpolating between two frames, which
# for a one-sided tail is the wrong thing anyway.
#
# Note p90 degenerates to the max for n<=8 (ceil(0.9*8)=8) — inherent to
# nearest-rank, and the reason `n_frames` is exposed.
FISH_LENGTH_ESTIMATE_VIEW_SQL = f"""
CREATE VIEW {FISH_LENGTH_ESTIMATE_VIEW_NAME} AS
WITH ranked AS (
    SELECT
        m.length_m   AS length_m,
        i.dive_id    AS dive_id,
        f.id         AS fish_id,
        f.name       AS model_name,
        f.species_id AS species_id,
        ROW_NUMBER() OVER (
            PARTITION BY f.id, i.dive_id ORDER BY m.length_m
        ) AS rn,
        COUNT(*) OVER (PARTITION BY f.id, i.dive_id) AS n
    FROM measurement m
    JOIN image i ON i.id = m.image_id
    JOIN fish f ON f.id = m.fish_id
    WHERE m.length_m IS NOT NULL
)
SELECT
    fish_id,
    dive_id,
    model_name,
    species_id,
    n AS n_frames,
    MAX(CASE WHEN rn = (9 * n + 9) / 10 THEN length_m END) AS length_p90_m,
    MAX(CASE WHEN rn = (n + 1) / 2      THEN length_m END) AS length_median_m,
    MAX(length_m) AS length_max_m,
    MIN(length_m) AS length_min_m,
    AVG(length_m) AS length_mean_m
FROM ranked
GROUP BY fish_id, dive_id, model_name, species_id, n
"""

DROP_FISH_LENGTH_ESTIMATE_VIEW_SQL = (
    f"DROP VIEW IF EXISTS {FISH_LENGTH_ESTIMATE_VIEW_NAME}"
)


# ---------------------------------------------------------------------------
# Fish-model species-mislabel suspects
# ---------------------------------------------------------------------------

FISH_MODEL_MISLABEL_SUSPECTS_VIEW_NAME = "fish_model_species_mislabel_suspects"

# A model's measured length is a strong prior on which model it is, so a frame
# whose length fits a *different* known model far better than its own label is
# a mislabel suspect. 18 real ones were found in prod on 2026-08-04 — 16 of
# them contiguous runs, i.e. a whole photo sequence of one model labeled as
# another.
#
# The asymmetry this view is built around: stage 14 measures the fish's
# PROJECTION (head and tail are back-projected at a single laser-derived
# depth), so an out-of-plane fish reads SHORT and never long. Hence:
#
#   * measured much LONGER than the label allows -> `high`. Foreshortening
#     cannot lengthen a fish, so geometry can't explain it.
#   * measured much SHORTER but matching another model -> `medium`. Genuinely
#     ambiguous: a Snook (455mm) angled ~21 deg reads 360mm, exactly Grouper.
#
# A group-maximum signal was tried and REMOVED after failing on real data:
# (a) Purple Angel 0.192 / Gray Anthias 0.195 / Yellow Anthias 0.200 are within
# 4%, so noise pushes a correct group's max into a neighbour's length and
# condemns the whole group; (b) one genuinely-mislabeled frame inflates its
# group's max and condemns the correct frames around it (prod dive 60: a single
# 601mm frame flagged all 19 real Groupers). Length can only discriminate
# models that differ by more than measurement noise plus foreshortening; where
# it can't, this view stays quiet rather than guessing.
#
# Deliberately NOT a filter on the accuracy view: a suspect frame is still a
# real measurement until a human re-labels it in Label Studio (which is
# authoritative — the hourly species sync overwrites the DB).
_MISLABEL_MIN_OWN_PCT_ERROR = 15.0
_MISLABEL_MAX_OTHER_PCT_ERROR = 10.0

FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL = f"""
CREATE VIEW {FISH_MODEL_MISLABEL_SUSPECTS_VIEW_NAME} AS
WITH frame_fit AS (
    SELECT a.image_id,
           r.name AS best_fit_model,
           100.0 * (a.length_m - r.known_length_m) / r.known_length_m
               AS best_fit_pct_error,
           ROW_NUMBER() OVER (
               PARTITION BY a.image_id
               ORDER BY ABS(a.length_m - r.known_length_m) / r.known_length_m
           ) AS rk
    FROM {FISH_MODEL_ACCURACY_VIEW_NAME} a
    CROSS JOIN fishmodelreference r
)
SELECT
    a.image_id,
    a.dive_id,
    a.model_name AS labeled_model,
    a.known_length_m,
    a.length_m,
    a.pct_error,
    f.best_fit_model,
    f.best_fit_pct_error,
    CASE
        WHEN a.pct_error > {_MISLABEL_MIN_OWN_PCT_ERROR} THEN 'high'
        ELSE 'medium'
    END AS confidence
FROM {FISH_MODEL_ACCURACY_VIEW_NAME} a
JOIN frame_fit f ON f.image_id = a.image_id AND f.rk = 1
WHERE f.best_fit_model <> a.model_name
  AND ABS(a.pct_error) > {_MISLABEL_MIN_OWN_PCT_ERROR}
  AND ABS(f.best_fit_pct_error) < {_MISLABEL_MAX_OTHER_PCT_ERROR}
"""

DROP_FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL = (
    f"DROP VIEW IF EXISTS {FISH_MODEL_MISLABEL_SUSPECTS_VIEW_NAME}"
)


# ---------------------------------------------------------------------------
# Every view, for the fresh-database bootstrap
# ---------------------------------------------------------------------------

# `(drop_sql, create_sql)` for each view, in creation order.
#
# Views are raw SQL owned by migrations, NOT part of `SQLModel.metadata`, so
# `create_all` cannot produce them. On a fresh database `run_alembic_upgrade`
# *stamps* head instead of upgrading (the historical migrations aren't
# idempotent against `create_all`), which used to leave every table present and
# every view missing — and because head was stamped, the next restart found
# nothing to do and it never self-healed. `database.create_all_views` replays
# this list on that path.
#
# Adding a view means adding it here as well as in its migration, or a fresh
# environment silently won't have it.
ALL_VIEW_DDL = (
    (DROP_DIVE_PIPELINE_STATUS_VIEW_SQL, DIVE_PIPELINE_STATUS_VIEW_SQL),
    (DROP_FISH_MODEL_ACCURACY_VIEW_SQL, FISH_MODEL_ACCURACY_VIEW_SQL),
    (DROP_FISH_LENGTH_ESTIMATE_VIEW_SQL, FISH_LENGTH_ESTIMATE_VIEW_SQL),
    (
        DROP_FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL,
        FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL,
    ),
)

ALL_VIEW_NAMES = (
    DIVE_PIPELINE_STATUS_VIEW_NAME,
    FISH_MODEL_ACCURACY_VIEW_NAME,
    FISH_LENGTH_ESTIMATE_VIEW_NAME,
    FISH_MODEL_MISLABEL_SUSPECTS_VIEW_NAME,
)
