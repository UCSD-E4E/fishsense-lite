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

DIVE_PIPELINE_STATUS_VIEW_NAME = "dive_pipeline_status"

# Slate-content marker mirrors `dive_controller.SLATE_CONTENT_MARKER`.
# Kept inline rather than imported to avoid `views.py` pulling controller
# imports during alembic migration runs (alembic env loads `views` to
# get this string before the FastAPI app is initialized).
_SLATE_CONTENT_MARKER = "Slate, Laser on slate"

# "Complete" everywhere = ≥1 completed-non-superseded row AND zero
# incomplete-non-superseded rows. Mirrors
# `get_dives_with_complete_laser_labeling`'s semantics so a dive with
# zero labels of a kind doesn't vacuously read as complete.
DIVE_PIPELINE_STATUS_VIEW_SQL = f"""
CREATE VIEW {DIVE_PIPELINE_STATUS_VIEW_NAME} AS
SELECT
    d.id AS dive_id,
    d.priority,
    d.dive_slate_id,

    -- Stage 0.1: every image in the dive has at least one LaserLabel
    -- row (any project, any state). The cohort selector is the
    -- inverse of this predicate.
    (EXISTS (SELECT 1 FROM image i WHERE i.dive_id = d.id)
     AND NOT EXISTS (
         SELECT 1 FROM image i
         WHERE i.dive_id = d.id
           AND NOT EXISTS (
               SELECT 1 FROM laserlabel ll WHERE ll.image_id = i.id
           )
     )) AS laser_preprocessed,

    -- Laser labeling: ≥1 completed-non-superseded AND zero
    -- incomplete-non-superseded.
    (EXISTS (
         SELECT 1 FROM laserlabel ll
         JOIN image i ON i.id = ll.image_id
         WHERE i.dive_id = d.id
           AND ll.superseded = FALSE
           AND ll.completed = TRUE
     )
     AND NOT EXISTS (
         SELECT 1 FROM laserlabel ll
         JOIN image i ON i.id = ll.image_id
         WHERE i.dive_id = d.id
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
         WHERE i.dive_id = d.id
           AND ll.completed = TRUE
           AND ll.superseded = FALSE
           AND ll.x IS NOT NULL
           AND ll.y IS NOT NULL
     )
     AND NOT EXISTS (
         SELECT 1 FROM laserlabel ll
         JOIN image i ON i.id = ll.image_id
         WHERE i.dive_id = d.id
           AND ll.completed = TRUE
           AND ll.superseded = FALSE
           AND ll.x IS NOT NULL
           AND ll.y IS NOT NULL
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
         WHERE i.dive_id = d.id
           AND htl.superseded = FALSE
           AND htl.completed = TRUE
     )
     AND NOT EXISTS (
         SELECT 1 FROM headtaillabel htl
         JOIN image i ON i.id = htl.image_id
         WHERE i.dive_id = d.id
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

    -- Stage 2: has PREDICTION clusters AND every image carrying a
    -- valid laser label (completed, not superseded, x/y both set) has
    -- a non-sentinel SpeciesLabel row. ≥1 such laser-valid image must
    -- exist (otherwise the predicate would be vacuously true for dives
    -- with no laser work yet — same convention as headtail_preprocessed).
    -- Cohort flipped 2026-05-05 from "every image" → "every laser-valid
    -- image" so species labeling now mirrors head/tail (per-image
    -- cascade from valid lasers) while keeping the cluster gate.
    (EXISTS (
         SELECT 1 FROM diveframecluster dfc
         WHERE dfc.dive_id = d.id
           AND dfc.data_source = 'PREDICTION'
     )
     AND EXISTS (
         SELECT 1 FROM laserlabel ll
         JOIN image i ON i.id = ll.image_id
         WHERE i.dive_id = d.id
           AND ll.completed = TRUE
           AND ll.superseded = FALSE
           AND ll.x IS NOT NULL
           AND ll.y IS NOT NULL
     )
     AND NOT EXISTS (
         SELECT 1 FROM laserlabel ll
         JOIN image i ON i.id = ll.image_id
         WHERE i.dive_id = d.id
           AND ll.completed = TRUE
           AND ll.superseded = FALSE
           AND ll.x IS NOT NULL
           AND ll.y IS NOT NULL
           AND NOT EXISTS (
               SELECT 1 FROM specieslabel sl
               WHERE sl.image_id = i.id
                 AND sl.label_studio_project_id IS NOT NULL
           )
     )) AS dive_images_preprocessed,

    -- Species labeling: ≥1 completed AND zero incomplete. (No
    -- superseded column on SpeciesLabel — the model doesn't carry
    -- one; the existing cohort selectors don't filter on it either.)
    (EXISTS (
         SELECT 1 FROM specieslabel sl
         JOIN image i ON i.id = sl.image_id
         WHERE i.dive_id = d.id
           AND sl.completed = TRUE
     )
     AND NOT EXISTS (
         SELECT 1 FROM specieslabel sl
         JOIN image i ON i.id = sl.image_id
         WHERE i.dive_id = d.id
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
         WHERE i.dive_id = d.id
           AND sl.content_of_image = '{_SLATE_CONTENT_MARKER}'
     )
     AND NOT EXISTS (
         SELECT 1 FROM specieslabel sl
         JOIN image i ON i.id = sl.image_id
         WHERE i.dive_id = d.id
           AND sl.content_of_image = '{_SLATE_CONTENT_MARKER}'
           AND NOT EXISTS (
               SELECT 1 FROM diveslatelabel dsl
               WHERE dsl.image_id = i.id
                 AND dsl.label_studio_project_id IS NOT NULL
           )
     )) AS slate_preprocessed,

    -- Slate labeling: ≥1 completed AND zero incomplete. (No
    -- superseded column on DiveSlateLabel — same as species.)
    (EXISTS (
         SELECT 1 FROM diveslatelabel dsl
         JOIN image i ON i.id = dsl.image_id
         WHERE i.dive_id = d.id
           AND dsl.completed = TRUE
     )
     AND NOT EXISTS (
         SELECT 1 FROM diveslatelabel dsl
         JOIN image i ON i.id = dsl.image_id
         WHERE i.dive_id = d.id
           AND (dsl.completed = FALSE OR dsl.completed IS NULL)
     )) AS slate_labeling_complete,

    -- Stage 13: dive has a LaserExtrinsics row.
    EXISTS (
        SELECT 1 FROM laserextrinsics le
        WHERE le.dive_id = d.id
    ) AS calibrated,

    -- Stage 14: ≥1 LABEL_STUDIO cluster AND zero LABEL_STUDIO clusters
    -- with fish_id NULL (every cluster bound to a fish via stage-14
    -- measurement).
    (EXISTS (
         SELECT 1 FROM diveframecluster dfc
         WHERE dfc.dive_id = d.id
           AND dfc.data_source = 'LABEL_STUDIO'
     )
     AND NOT EXISTS (
         SELECT 1 FROM diveframecluster dfc
         WHERE dfc.dive_id = d.id
           AND dfc.data_source = 'LABEL_STUDIO'
           AND dfc.fish_id IS NULL
     )) AS measured

FROM dive d
"""

DROP_DIVE_PIPELINE_STATUS_VIEW_SQL = (
    f"DROP VIEW IF EXISTS {DIVE_PIPELINE_STATUS_VIEW_NAME}"
)
