# Image → Fish Length Pipeline

End-to-end walk-through of how a raw `.ORF` capture becomes a row in the
`Measurement` table. The pipeline is split across the two workflow
workers and depends on human-in-the-loop labeling in Label Studio at
several points.

Two parallel arms run before a dive can be measured:

* **Laser calibration arm** — per-camera, run once before measuring any
  dive shot with that camera.
* **Dive imagery arms** — per-dive (species + head/tail keypoints, plus
  the dive slate that ties the dive to a calibration).

Both feed into the final measurement step.

For the source-of-truth port status table, see the "Notebook port
status" section of [CLAUDE.md](../CLAUDE.md).

## A. Laser calibration arm (per camera)

| Stage | Notebook | Worker | Status |
|---|---|---|---|
| 0.1 | `preprocess_laser_images` | data | ported (hourly self-paced) |
| 0.3 | `populate_label_studio_project` (Laser) | api | ported |
| —   | hourly `SyncLabelStudioLaserLabelsWorkflow` | api | scheduled |
| 13  | `perform_laser_calibration` | data | ported (kernel in fishsense-core) |

1. **Stage 0.1** rectifies + JPEG-encodes laser-only flat-target shots
   (data-worker, writes to `preprocess_jpeg/`). Hourly schedule
   self-picks the next HIGH-priority dive without `LaserExtrinsics`.
2. **Stage 0.3** `PopulateLaserLabelStudioProjectWorkflow` pushes them
   into the LS Laser project.
3. *Humans label the laser dot* in each frame.
4. The hourly `SyncLabelStudioLaserLabelsWorkflow` pulls labels back
   into Postgres as `LaserLabel` rows.
5. **Stage 13** (`PerformLaserCalibrationWorkflow(dive_id)` on the
   data-worker) runs the Atanasov fit on all collected `LaserLabel`s
   and produces a `LaserExtrinsics` row tying the laser line to the
   camera intrinsics. The fit delegates to
   `fishsense_core.laser.calibrate_laser`; numerical equivalence vs
   the pre-refactor inline implementation is validated against prod
   (max axis delta 0.011°, max position delta 390µm over 8 dives;
   see [scripts/validate_stage13_refactor.py](../services/fishsense-data-processing-workflow-worker/scripts/validate_stage13_refactor.py)).

## B. Dive species arm (per dive)

| Stage | Notebook | Worker | Status |
|---|---|---|---|
| 1   | `cluster_dive_frames` | data | ported |
| 2   | `preprocess_dive_images` | data | ported |
| 4   | `populate_label_studio_project` (Species) | api | ported |
| 4.2 | `sync_species_labels` | api | partial — generic sync runs; species-specific TBD |

6. **Stage 1** clusters consecutive frames of the same fish (data-worker).
7. **Stage 2** rectifies + overlays + JPEG-encodes dive frames into
   `preprocess_groups_jpeg/` (data-worker).
8. **Stage 4** `PopulateSpeciesLabelStudioProjectWorkflow` pushes them
   into the LS Species project.
9. *Humans bounding-box and species-tag each fish.*
10. **Stage 4.2** sync pulls species labels back. (`SyncLabelStudioSpeciesLabelsWorkflow`
    is registered but the species-specific reconciliation is not yet
    fully ported — see CLAUDE.md.)

## C. Head/tail keypoints (per fish crop)

| Stage | Notebook | Worker | Status |
|---|---|---|---|
| 5.1 | `preprocess_headtail_images` | data | ported |
| 5.3 | `populate_label_studio_project` (HeadTail) | api | ported |
| —   | hourly `SyncLabelStudioHeadTailLabelsWorkflow` | api | scheduled |

11. **Stage 5.1** crops around each species bbox, rectifies the crop,
    JPEG-encodes to `preprocess_headtail_jpeg/` (data-worker).
12. **Stage 5.3** `PopulateHeadTailLabelStudioProjectWorkflow` pushes
    crops into the LS HeadTail project.
13. *Humans drop a head + tail keypoint on each fish.*
14. The hourly `SyncLabelStudioHeadTailLabelsWorkflow` pulls
    `HeadTailLabel` rows back into Postgres.

## D. Dive slate (which calibration applies to this dive)

| Stage | Notebook | Worker | Status |
|---|---|---|---|
| 9   | `preprocess_slate_images` | data | ported |
| 11  | `populate_label_studio_project` (DiveSlate) | api | ported |
| —   | hourly `SyncLabelStudioDiveSlateLabelsWorkflow` | api | scheduled |
| 12  | `sync_slate_label` | api | ported |

15. **Stage 9** renders PDF-composited slate JPEGs (data-worker, writes
    to `preprocess_slate_images_jpeg/`).
16. **Stage 11** `PopulateDiveSlateLabelStudioProjectWorkflow` pushes
    slate images into LS.
17. *Humans transcribe slate metadata, including which laser
    calibration applies.*
18. The hourly `SyncLabelStudioDiveSlateLabelsWorkflow` pulls
    `DiveSlateLabel` rows back into Postgres. The LS image is a
    composite (PDF panel on the left, photo on the right), so the
    sync activity opens the slate PDF via the file-exchange and
    shifts reference-point + slate-rectangle x-coords left by the
    rendered panel width to land them in photo-frame coords.

## E. Measure

| Stage | Notebook | Worker | Status |
|---|---|---|---|
| 6.1 | `update_dive_image_groups` | api | not started |
| 14  | `measure_fish` | data | ported (kernel in fishsense-core) |

19. **Stage 6.1** reconciles species labels back into the frame clusters
    from stage 1.
20. **Stage 14** (`MeasureFishWorkflow(dive_id)` on the data-worker)
    computes the fish length per top-three species label:
    1. Laser-dot pixel + `LaserExtrinsics` → triangulate depth at the
       laser-hit point (`compute_world_point_from_laser`).
    2. Feed that depth into `compute_world_point_from_depth` for the
       head and tail pixels — kernel is `K⁻¹ · [x, y, 1] · depth`,
       positive sign convention, no internal flip.
    3. `‖head₃d − tail₃d‖` is the fish length, written to
       `Measurement`.

    Returns a `MeasureFishResult` summary (`measured`,
    `dropped_nan`, `missing_laser_or_headtail`, `missing_cluster`)
    so silent drops from the notebook are observable. Raises if
    `LaserExtrinsics` is absent — run stage 13 first.

The math layer for stages 13/14 is pinned down by synthetic-geometry
tests (commits `15a545a`, `a5e92c6`); the open follow-up is a
real-frame regression vs historical `Measurement` rows. See the
"stage14 sign-flip" section of [CLAUDE.md](../CLAUDE.md) for details
and current status.

## State of the port today

* **Ported and runnable:** all four data-worker preprocess stages
  (0.1, 2, 5.1, 9), all four `Populate*LabelStudioProjectWorkflow`s,
  the three sync-back workflows scheduled hourly
  (`SyncLabelStudioLaserLabelsWorkflow`,
  `SyncLabelStudioHeadTailLabelsWorkflow`,
  `SyncLabelStudioDiveSlateLabelsWorkflow`), the dashboard-config
  writer, and the laser-calibration + measurement workflows
  (`PerformLaserCalibrationWorkflow`, `MeasureFishWorkflow` —
  both data-worker, on-demand).
* **Still needed before a fresh dive can be measured end-to-end in
  the monorepo:** stage 6.1 (cluster reconciliation), plus the
  real-frame regression for stages 13 + 14 against historical
  `Measurement` rows (gated on prod-DB / api-worker access — see
  CLAUDE.md).

## Workflow scheduling vs on-demand

Of the workflows registered with the api-worker, four are on a
Temporal schedule (all hourly):

* `SyncLabelStudioLaserLabelsWorkflow`
* `SyncLabelStudioHeadTailLabelsWorkflow`
* `SyncLabelStudioDiveSlateLabelsWorkflow`
* `UpdateDashboardConfigWorkflow`

The eight `Create*` and `Populate*` workflows are **on-demand**:

* `Create*` is bootstrap-only and idempotent — once the LS project
  exists, re-running just returns its ID.
* `Populate*` requires a `dive_id` argument; a clock-driven schedule
  has no way to choose one.

See [worker.py](../services/fishsense-api-workflow-worker/src/fishsense_api_workflow_worker/worker.py)
for the registration list and `schedule_workflows` body.

## File-exchange URL contract (data flow between workers)

```
GET  /api/v1/exchange/raw/{checksum}.ORF             # api-worker upload, data-worker reads
GET  /api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf # api-worker upload, data-worker reads (stage 9)
PUT  /api/v1/exchange/{folder}/{checksum}.JPG        # data-worker writes preprocessed JPEGs
```

Output folders (matching nginx GET routes used by the labeler frontend):

| Stage | Folder |
|---|---|
| 0.1 | `preprocess_jpeg` |
| 2   | `preprocess_groups_jpeg` |
| 5.1 | `preprocess_headtail_jpeg` |
| 9   | `preprocess_slate_images_jpeg` |
