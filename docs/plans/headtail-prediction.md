# Plan — model-assisted head/tail labeling (stage 5.1 pre-annotations)

Status: **proposed, not started.**

Seed the per-dive head/tail Label Studio projects with model-predicted
`Snout` / `Fork` keypoints, the way `LaserPrediction` already seeds the laser
ones: a SAM3 mask of the fish the laser is on, keypointed by
`fishsense_core`'s `FishHeadTailDetector`.

Read §0 first. The measurement that justifies this also corrects the metric
you would naturally reach for, and the correction is the difference between
"this fails" and "this works".

---

## 0. Evidence — measured 2026-09-02, n=560

Tool: [`tools/validate_headtail_predictions.py`](../../tools/validate_headtail_predictions.py)
(`manifest` / `predict` / `report` / `visualize`; read-only, credentials from
the environment). Oracle: every image carrying both a completed
non-superseded `HeadTailLabel` and a valid `LaserLabel` — **16,987 images
across 151 dives** in the 2026-09-02 prod backup. Sampled 4 per dive, evenly
spaced within each dive rather than head-of-list, so a dive's setup frames
don't stand in for it.

### 0.1 Score on **length error**, not orientation

`find_head_tail_img` gets head-vs-tail identity right only **74.3%** of the
time, and that does not matter here. `laser_geometry.measure_length_at_depth`
places **both** keypoints at the same laser depth and returns
`‖head3d − tail3d‖`, which is exactly symmetric; every other reference to
`head_x`/`tail_x` in the tree is a NOT NULL existence check
(`views.py`, `dive_cohort_controller`, `measure_fish_activity`'s guard).
**Nothing in fishsense-lite reads which point is the snout.** A swapped
pre-annotation costs a reviewing labeler a glance.

This is worth stating explicitly because the first pass of this analysis
scored orientation, called the result a failure, and was wrong. Do not
re-derive the objection.

### 0.2 What the length error actually is

Of 560 labelled images, **57.7% get a prediction**; 24.8% produce no detection
and 17.0% have no fish under the laser dot. Both of those are abstentions, and
abstaining is the correct behaviour, not a defect.

These are the numbers for a **Fishial full-frame** mask source, which is what
this section measured and what the plan originally assumed. §0.2b changes that
backend and lifts coverage to ~75%; the error distribution below is
substantially unchanged by the swap, so it still stands as the quality
estimate.

Of the 323 predictions:

| \|length error\| | share of predictions |
|---|---|
| ≤ 2% | 36.5% |
| ≤ 3% | 46.7% |
| ≤ 5% | **63.5%** |
| ≤ 10% | 80.8% |

p50 **3.2%**, p75 8.2%, p90 17.1%, p95 26.2%. Signed median **+0.4%** — so
there is *no* systematic caudal-tip-instead-of-fork bias, which was the
a-priori risk given that every reference length in `fishmodelreference` is a
fork length.

Judge this against the budget the pipeline already carries: per-dive
calibration scale error runs −8…+4%, and lengths are read at p90 (see the
`project-fish-model-measurement-validation` notes). A median of 3.2% sits
inside the existing noise.

### 0.2b The mask backend: SAM3 on a laser-centred crop

The 42.3% that get no prediction are not "images without a fish" — they are
images where the detector did not find the fish the laser is on. Both
abstention modes have near-identical target-size distributions (p50 87 px and
90 px at model resolution, against 144 px for successes), so they are one
phenomenon: the fish is too small once a 4014×3016 frame is squeezed to the
1058×795 the Mask R-CNN export accepts.

`find_head_tail_img` runs on whatever binary mask it is handed, so the mask
source is a free parameter. Five options, all scored on the same 80 frames
against the same human labels:

| backend | predicted | snout p50 | fork p50 | \|len\| p50 | **usable** (≤5% len err, of all 80) | s/frame | queue |
|---|---|---|---|---|---|---|---|
| Fishial, full frame | 40/80 | 3.6% | 10.7% | 5.7% | 19/80 = 23.8% | 2.9 | CPU |
| Fishial, laser tile | 51/80 | 4.6% | 10.7% | 11.3% | 18/80 = 22.5% | 1.7 | CPU |
| SAM3, full frame | 44/80 | 1.9% | 9.7% | 6.3% | 18/80 = 22.5% | 0.9 | GPU |
| SAM3, tiled (~20 tiles) | 58/80 | 1.4% | 9.6% | 6.0% | 27/80 = 33.8% | 11.3 | GPU |
| **SAM3, laser crop** | **62/80** | **1.3%** | 9.2% | 6.4% | **35.0–51.4%** | **0.5** | GPU |

"Usable" is the product metric — a pre-annotation within 5% of the labeler's
own length, per *labelled image*, not per prediction. **SAM3 on a laser-centred
1800×1350 crop wins on every axis at once**, including cost: 47% more usable
predictions than the Fishial full-frame design this plan originally assumed,
and the cheapest option measured.

Three things that table says which are easy to get wrong:

* **Resolution is the lever, not the detector.** SAM3 full-frame gains nothing
  over Fishial full-frame (22.5% vs 23.8%) despite finding four more fish. Swap
  the model without fixing resolution and you have spent a GPU for nothing.
* **Detections are not predictions.** Fishial's laser tile finds 11 more fish
  than its full frame and yields *fewer* usable ones — the extra fish are the
  marginal ones, and their keypoints are poor. Only SAM3 converts extra recall
  into usable output (47% conversion, against 35% for Fishial tiled).
* **The laser makes tiling unnecessary.** A tile sweep searches the whole frame
  for a fish whose position is already known. One crop centred on the dot
  reaches the same resolution for a single inference instead of ~20 — 0.6 s
  against 11.3 s — and scores slightly *better*, because the fish is centred
  and never split across a tile boundary. The gate is not just a filter; it
  tells the predictor where to look.

**The fork is the one thing no backend fixes** — 9.6–10.7% across all five. It
is a property of `find_head_tail_img` returning a mask extremity, not of the
mask source, and it is what drives the length tail. Attack it there or not at
all.

**Note the landscape-only bug** (`fish-detection-labeling.md` §0.5): while it
mattered for Fishial tiling, the chosen design crops 4:3 and is unaffected.

### 0.2c Crop size: tuned, then validated on held-out frames

The window was tuned against the same human labels, on the 80-frame set, and
then the candidates were re-run on **70 frames from 63 other dives with zero
overlap**. Both numbers are given because the difference between them is the
point:

| crop | scale | tuning (n=80) | **held-out (n=70)** |
|---|---|---|---|
| 1000×750 | 1.01 | 30.0% | — |
| 1400×1050 | 0.72 | 36.2% | 47.1% |
| **1800×1350** | 0.56 | 35.0% | **51.4%** |
| 2200×1650 | 0.46 | 35.0% | 45.7% |
| 3000×2250 | 0.34 | **38.8%** | 38.6% *(worst)* |

**3000×2250 topped the tuning set and came last on held-out.** With ~5
candidates at n=80 and a standard error near 5 points, the tuning "winner" was
noise, and picking it would have shipped the worst option. That is the entire
reason the held-out set was rectified before the sweep ran rather than after.

**1800×1350** is the only size that looks good on both — best detection rate on
the tuning set (62/80), best usable rate and best fork error on held-out — and
it sits where the theory puts it, between two failure modes that are visible in
the table. Too small and the crop cuts the fish: 1000×750 contains both human
keypoints for only 90% of frames, and its fork error is the worst measured
(13.1%). Too large and resolution is given back: 3000×2250 finds the fewest
fish (53/70).

Read the choice as "anywhere from 1400 to 2200 is fine, and 1800 is the middle
of that plateau" rather than as a precise optimum. Snout error is flat at
1.3–1.7% across every size — it is insensitive to this parameter entirely.

### 0.3 The laser gate is load-bearing

It picked a different fish than "largest detection" would have on **65 of
323** predictions (20%), and abstained on **95 images** (17%) where no instance
sat under the dot. The gate survives the backend change intact — Fishial
returns an instance *label map*, so the test is one array lookup
(`labels[round(y), round(x)]`); SAM3 returns per-instance masks, so it is the
first mask whose pixel at the dot is set. Either way it is a lookup, not a
search.

Under the chosen backend the gate does more than filter: it also supplies the
crop centre (§0.2b), which is what removes the need for a tile sweep.

### 0.4 What is weak

Fork localization: p50 **7.4%** of fish length, against **2.8%** for the
snout. That asymmetry is what drives the length tail. Watch p90, not the
median.

### 0.5 A cheap confidence signal exists

Mask area / `pred_len²` — a fish silhouette runs ~0.15–0.30 — restricted to
**0.18–0.32** keeps 76% of predictions and moves p90 from 17.1% → **12.7%**,
within-5% from 63.5% → 69%. Apparent size and scene clutter (`n_instances`)
were both tested and are flat; do not bother with them.

### 0.6 What I verified, and what I did not

**Verified.** That local rectification is a sound substitute for the archived
frame: `RectifiedImage(RawImage(bytes), intrinsics)` reproduces the stage-5.1
Garage JPEG **byte-for-byte** (max pixel difference 0) on one image from each
of the 11 dives where both exist. This mattered because only **350 of the
16,987** oracle images still have a JPEG in Garage — the head/tail corpus
predates both stage 5.1 and the Garage migration.

**Verified.** `FishSegmentation.inference` wants **BGR** — i.e.
`RectifiedImage.data` as-is. Feeding it RGB measurably degrades detection.
("RGB grid space" in `predict_keypoint_depths` means demosaiced-vs-Bayer, not
channel order.)

**Not verified.** Whether pre-annotations actually speed labelers up. That is
a human-factors question this corpus cannot answer, and it is the real
success criterion — see §8.

**Not verified.** Behaviour on dives shot after 2026-09-02, and on any camera
or site not represented in the 151 dives sampled.

**Limits of the §0.2b backend comparison specifically.** It is **n=80**, so the
usable-rate differences carry roughly ±10 points of sampling error — the
ordering is clear and repeated across four independent measurements, but treat
the exact percentages as indicative. The crop size was tuned separately
(§0.2c). SAM3's OOM losses were an artefact of a 6 GB laptop card and are
excluded from the reported rates. And every number is from a *single* SAM3
prompt, `["fish"]`; the coral-gardeners pipeline defaults to
`["fish", "small fish"]`, which was not evaluated here.

**A known floor.** Human head/tail labels carry their own error, so the
reported numbers are prediction error *plus* label noise. The floor is not
zero.

---

## 1. Design decisions

### 1.1 Read the stage-5.1 JPEG, not the raw `.ORF`

The laser and slate predict parents stage raw bytes from the NAS into Garage
scratch, because they run *before* any JPEG exists. Head/tail is different:
stage 5.1 has already written `preprocess_headtail_jpeg/{checksum}.JPG`, and
that JPEG is exactly what the validation ran on and exactly what the labeler
will see.

Consequences, all good:

* No `stage_raw_bytes_for_dive_activity`, no `cleanup_raw_bytes_for_dive_activity`,
  no NAS traffic at all.
* ~2 s/image instead of ~10 s — the rawpy decode dominates the raw path.
* The predictor cannot get ahead of the JPEG, so the JPEG-presence gate that
  laser populate needs (`_gate_on_jpeg_presence`) is unnecessary here:
  prediction-gating subsumes it.

Cost: the data-worker's object-store client gains a **read** of the JPEG
prefix it currently only writes. That is a deliberate widening of the
asymmetry described in CLAUDE.md's key-contract section — the api-worker
stages and deletes scratch, the data-worker reads scratch and writes JPEGs.
It is mild (the worker is reading its own output, in its own bucket) but it
is a boundary change and belongs in review, not in a helper.

### 1.2 GPU queue — a reversal, and the one real cost

An earlier draft put this on the CPU queue, because `fishsense_core.fish` is in
the base wheel with ONNX Runtime statically linked and needs no checkpoint. The
§0.2b benchmark overturns that: SAM3 is a torch model, so the stage registers
in the **`gpu` role** on `fishsense_data_processing_gpu_queue` and inherits the
GPU/CPU-fallback machinery (`ensure_gpu_worker_running_activity`, the
annotation-based failure counter, the `unavailable` early return).

This is the only thing the backend choice costs, and it is worth naming
plainly. Against it: at 0.6 s/frame the stage is *cheaper* than the 2.9 s CPU
design it replaces, so the GPU is held briefly per image rather than
continuously; and the GPU queue already means "prefer a GPU", not "require
one" — the CPU-fallback Deployment serves the same checkpoints, so a GPU
shortage degrades speed, not availability.

Two consequences to carry through:

* Both predict parents already **return before staging any bytes** when
  `ensure_gpu_worker_running_activity` reports `unavailable`. This parent must
  do the same; the dive stays in the cohort for the next firing.
* SAM3's weights are ~3.3 GB from HuggingFace. Unlike Fishial's embedded model
  they must be baked into the data-worker image or mounted, the way
  `predict_laser_image`'s checkpoint is. That is a Dockerfile change, and it is
  a prerequisite, not a detail.

### 1.3 Gate on the laser, and record why an image was skipped

Take the instance id under the image's validated `LaserLabel`; if it is 0,
abstain. An image may carry more than one valid laser label (461 prod images
do) — try each and take the first that lands on a fish, which is what the
validation measured.

Abstentions are distinguishable in the row, the way `LaserPrediction`
separates `rejected_out_of_region` from an ordinary non-detection: without
that, "the model found nothing" and "the gate rejected it" look identical and
a mis-sized gate reads as a model that stopped working.

### 1.4 Confidence gate: record it, apply it at seed time

Store `mask_area_px` and the silhouette ratio on the row and set
`rejected_low_confidence` when it falls outside the configured band, defaulting
to §0.5's 0.18–0.32. Keep the row either way. Applying the gate at seed time
rather than predict time means the band can be retuned from data already
collected, without a re-predict pass.

Explicitly **not** gated on: orientation (§0.1), apparent size, or
`n_instances` (§0.5).

### 1.5 Decouple headtail populate into its own parent

Today `PreprocessHeadtailImagesParentWorkflow` dispatches
`PopulateHeadTailLabelStudioProjectWorkflow` as a child after cleanup. Predict
has to run *between* those two, so populate moves to its own schedule — the
same restructuring already done for laser (+10 predict, +12 populate) and
species (+20).

This is not optional plumbing. The predict cohort requires "no `HeadTailLabel`
row", and populate seeds sentinel rows; populating before predicting would
permanently starve the image of a prediction. That is precisely the trap
`_select_unlabeled_images` documents on the laser side, and the fix is the
same: **populate is prediction-gated.**

### 1.6 Predictions never count toward the human gates

`HeadTailPrediction` is its own table, like `LaserPrediction`. No cohort
selector, no `dive_pipeline_status` column, and no measurement path may read
it. A labeler's confirmation still lands as an ordinary `HeadTailLabel` via
the existing sync.

---

## 2. API surface (`services/fishsense-api`)

### 2.1 Model — `models/head_tail_prediction.py`

```python
class HeadTailPrediction(ModelBase, table=True):
    __table_args__ = (UniqueConstraint("image_id", name="uq_headtail_prediction_image"),)

    id: int | None = Field(default=None, primary_key=True)
    # None when nothing was predicted; the status column says why.
    head_x: float | None = None
    head_y: float | None = None
    tail_x: float | None = None
    tail_y: float | None = None
    # Rectified frame dims the coords are relative to — populate needs them
    # to convert pixels to Label Studio keypoint percentages.
    width: int | None = None
    height: int | None = None
    # Which fish, and how fish-shaped it was. `silhouette_ratio` is
    # mask_area_px / pred_len_px**2; see §0.5.
    mask_area_px: int | None = None
    silhouette_ratio: float | None = None
    # Origin of the laser-centred crop the mask was found in (§4.3). The
    # keypoints above are already lifted back to rectified-frame pixels; this
    # is kept so a suspect prediction can be re-examined in the exact window
    # the model saw, and so a change to the crop size is visible in the data
    # rather than only in the code that produced it.
    crop_x: int | None = None
    crop_y: int | None = None
    # The laser label that selected the instance — provenance, and what makes
    # a re-predict after a laser is superseded a drainable cohort rather than
    # a hand-run backfill (the LaserDepth lesson).
    laser_label_id: int | None = Field(default=None, foreign_key="laserlabel.id")
    # "predicted" | "no_detections" | "laser_off_all_fish" | "headtail_failed"
    status: str = Field(default="predicted")
    rejected_low_confidence: bool = Field(default=False)
    created_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)
    image_id: int | None = Field(default=None, foreign_key="image.id")
```

`laser_label_id` is the one field worth arguing about, and it is there for the
reason `LaserDepth` carries its inputs: cohorts that select on *mismatch*
rather than absence drain on their own after a recalibration or a supersede
pass. Without it, a prediction made from a laser label that RANSAC later
supersedes is silently stale forever.

Register it in `database.py` (the model registry) or it will not appear in
autogenerated migrations.

### 2.2 Endpoints — `controllers/head_tail_prediction_controller.py`

| Route | Returns |
|---|---|
| `GET /api/v1/images/{image_id}/headtail-prediction/` | `HeadTailPrediction \| None` |
| `PUT /api/v1/images/{image_id}/headtail-prediction/` | upsert on the natural key |
| `GET /api/v1/dives/{dive_id}/headtail-predictions/` | `List[HeadTailPrediction]` |

The PUT **must** resolve the natural key and merge, not blind-upsert — the
2026-07-21 lesson from all four `put_*_label` handlers, which 500'd on
duplicate-key the moment populate retried. Register the controller in
`controllers/__init__.py` or the routes silently never bind.

Watch route ordering: per the `project-fastapi-route-order-across-modules`
note, `/dives/{dive_id}/...` collides with `/dives/select-next/...` if the
import order in `controllers/__init__.py` changes. Add the import at the end.

### 2.3 SDK — `libs/fishsense-api-sdk`

`images.get_headtail_prediction` / `put_headtail_prediction` /
`get_headtail_predictions`, plus the Pydantic mirror in
`models/head_tail_prediction.py`. `test_sdk_drift.py` parametrizes every
paired model, so the mirror must match field-for-field in the same PR.

### 2.4 Migration

One alembic revision adding the table + `uq_headtail_prediction_image`.
Nothing to backfill: an absent row reads as "not predicted yet" and the cohort
picks it up.

---

## 3. Contracts (`libs/fishsense-shared/preprocess_contracts.py`)

Cross-worker DTOs, so they live beside the existing ones:

```python
class PredictHeadtailImagesInput:      # api-worker -> data-worker
    dive_id: int
    image_checksums: list[str]
    image_ids: list[int]
    laser_points: list[list[list[float]]]   # per image, [[x, y], ...]
    laser_label_ids: list[list[int]]        # parallel to laser_points
    jpeg_folder: str                        # "preprocess_headtail_jpeg"

class HeadtailPredictionResult:        # data-worker -> api-worker
    image_id: int
    # Rectified-frame pixels — already lifted out of the crop (§4.3).
    head_x: float | None
    head_y: float | None
    tail_x: float | None
    tail_y: float | None
    width: int | None
    height: int | None
    mask_area_px: int | None
    silhouette_ratio: float | None
    crop_x: int | None
    crop_y: int | None
    laser_label_id: int | None
    status: str
```

`jpeg_folder` is a parameter rather than a constant for the same reason the
preprocess DTOs carry theirs — the physical key contract belongs to
`fishsense_shared.object_store`, and the resolver names the folder so the
data-worker never hard-codes a prefix.

---

## 4. Workflow + activity breakdown

```
PredictHeadtailImagesParentWorkflow          (api-worker, fishsense_api_queue, hourly +32)
  1. select_next_high_priority_dive_for_headtail_prediction_activity -> dive_id | None
  2. resolve_headtail_predict_inputs_activity(dive_id) -> PredictHeadtailImagesInput
  3. ensure_gpu_worker_running_activity -> gpu | cpu_fallback | unavailable
       on `unavailable`: RETURN NOW (§1.2) — nothing has been staged, and the
       dive stays in the cohort for the next firing
  4. start_child_workflow PredictHeadtailImagesWorkflow
       (fishsense_data_processing_gpu_queue)
  5. persist_headtail_predictions_activity(results) -> int
```

No `stage_raw` and no `cleanup_raw` (§1.1) — this parent is structurally
closer to stages 13/14 than to the preprocess parents, and lighter still than
the other two predict parents, which do stage raw bytes.

### 4.1 `select_next_high_priority_dive_for_headtail_prediction_activity`

Cohort: `Priority.HIGH` **and** at least one canonical image with

* a *valid* `LaserLabel` (`completed`, `superseded=False`, `x`/`y` set), and
* **no** non-sentinel `HeadTailLabel` row in any project, and
* **no** `HeadTailPrediction` row naming one of that image's still-valid laser
  labels.

The third clause is the mismatch-not-absence form from §2.1. Correlate every
subquery explicitly with `.correlate(...)` — auto-correlation only reaches the
immediately enclosing SELECT, and an uncorrelated one compiles to a
multi-row scalar subquery that Postgres rejects while SQLite silently answers
with the first row. That exact bug 500'd two selectors on every hourly poll on
2026-08-18. Add the selector to `test_api_postgres_integration.py`.

### 4.2 `resolve_headtail_predict_inputs_activity(dive_id)`

Returns the DTO. Filters images on the same predicate as the selector — the
resolver and the cohort must agree, or the fan-out does work the cohort never
promised. Emits each image's valid laser points **and** their label ids.

### 4.3 `predict_headtail_image` (data-worker, `gpu` role)

```
download_processed_jpeg(folder, checksum)          # see §1.1
  -> decode, then CROP 1800x1350 centred on the laser dot,
     clamped to the frame                          # §0.2b: the gate says where to look
  -> SAM3 concept prompt ["fish"] on the crop      # one inference, not a tile sweep
  -> the returned instance mask covering the dot   (else abstain)
  -> FishHeadTailDetector.find_head_tail_img(mask)
  -> lift keypoints back by the crop offset
  -> HeadtailPredictionResult
```

The crop window is **1800×1350** (§0.2c); it is ~45% of the frame each way, so
the fish arrives at roughly 2.2× the resolution a full-frame pass would give it. Clamp the origin rather than padding, so the
window is always full-size and the model never sees a letterboxed edge.

**Lift the keypoints back by `(ox, oy)` before returning them.** They come out
of the detector in crop coordinates, and `HeadTailPrediction` is defined in
rectified-frame pixels — the same space as `LaserLabel.x/y` and the labeler's
own clicks. Getting this wrong is a silent, plausible-looking offset, so it is
worth a dedicated test (§7.11).

Model loading is process-wide behind a double-checked lock, exactly as
`predict_laser_image._get_detector` does it — activities run in a real
`ThreadPoolExecutor`, so on a cold pod the whole first batch enters together
and an unguarded load gives every thread its own copy of the session. For SAM3
that also matters for VRAM: a 6 GB card OOM'd on ~9% of frames during the
benchmark with one session, and per-frame `torch.cuda.empty_cache()` plus
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` were needed to get through
an 80-frame sweep. Size `max_concurrent_activities` accordingly.

Inference runs off-loop via `asyncio.to_thread`.

### 4.4 `PredictHeadtailImagesWorkflow` (data-worker)

Thin fan-out over `predict_headtail_image`, returning
`list[HeadtailPredictionResult]`. No SDK calls, no NAS, every decision already
in the DTO.

### 4.5 `persist_headtail_predictions_activity`

Upserts each result via the SDK, on the api side so the call stays on the
interior network. Direct copy of `persist_laser_predictions_activity`.

### 4.6 Populate changes

`populate_headtail_label_studio_project_activity` gains, mirroring the laser
version:

* `_prediction_annotations(prediction)` — emits **two** `kp-1` keypoints,
  `Snout` and `Fork`, as percentages of the recorded `width`/`height`. Skips a
  prediction whose coords are None or whose `rejected_low_confidence` is set.
* `_select_unlabeled_images(..., predicted_image_ids)` — prediction-gated
  (§1.5).

The `from_name` is `kp-1` and the labels are `Snout` / `Fork`; both must match
`HEADTAIL_LABELING_CONFIG_XML` exactly, since the sync activity filters
annotations on `r["from_name"] == "kp-1"`.

### 4.7 Backfill for already-populated dives

Every dive whose head/tail project already has tasks will never be re-populated,
so it would never receive pre-annotations — the gap that made the slate
detector's first firing seed 0/28 on dive 65. Reuse that fix:
`backfill_headtail_predictions_for_dive_activity` attaches predictions to
*existing* LS tasks via `ls.predictions.create`, idempotently, called by the
predict parent after persist, plus a `BackfillHeadtailPredictionsWorkflow(dive_id)`
for catch-up.

---

## 5. Schedule

| Slot | Workflow |
|---|---|
| +30 | `PreprocessHeadtailImagesParentWorkflow` (existing; **stops dispatching populate**) |
| **+32** | `PredictHeadtailImagesParentWorkflow` (new) |
| **+34** | `PopulateHeadTailLabelStudioProjectParentWorkflow` (new parent, existing child) |

Hourly, `ScheduleOverlapPolicy.SKIP`, mirroring the laser +10/+12 pair. The
2-minute gap is the established spacing; if predict overruns, populate simply
seeds nothing that hour and the next firing picks it up — the same
self-correcting behaviour laser has. `test_schedule_registration.py` pins the
stagger and must be updated.

Operator action at deploy: nothing to delete. The existing headtail preprocess
schedule stays; only its populate dispatch goes away.

---

## 6. Files to add / change

**fishsense-api** — `models/head_tail_prediction.py`,
`controllers/head_tail_prediction_controller.py`, register in
`controllers/__init__.py` and `database.py`, one alembic revision.

**fishsense-api-sdk** — `models/head_tail_prediction.py`, three client methods
on the images client.

**fishsense-shared** — `preprocess_contracts.py` (+2 DTOs).

**fishsense-data-processing-workflow-worker** —
`activities/predict_headtail_image.py`,
`workflows/predict_headtail_images_workflow.py`,
`object_store.py` (+`download_processed_jpeg`), register in `roles.py` under
**`gpu`** (§1.2), and bake the SAM3 weights into the data-worker image.

**fishsense-api-workflow-worker** —
`activities/select_next_high_priority_dive_for_headtail_prediction_activity.py`,
`activities/resolve_headtail_predict_inputs_activity.py`,
`activities/persist_headtail_predictions_activity.py`,
`activities/backfill_headtail_predictions_activity.py`,
`workflows/predict_headtail_images_parent_workflow.py`,
`workflows/populate_headtail_labels_parent_workflow.py`,
edits to `populate_headtail_label_studio_project_activity.py`,
`preprocess_headtail_images_parent_workflow.py` (drop the populate dispatch),
`worker.py` (two schedules).

---

## 7. Test list — failing first, in order

TDD is mandatory here; the data-worker activity gets the full four-test
structure.

**fishsense-api**
1. `PUT` creates, `PUT` again updates the same row (natural-key upsert, no
   duplicate-key 500).
2. `GET` by image returns None when absent.
3. `GET` by dive returns only that dive's rows.
4. `test_sdk_drift` passes for the new paired model.
5. Migration up/down against SQLite.

**Cohort selector**
6. Dive with a valid laser + no headtail label + no prediction → selected.
7. Dive whose image has a non-sentinel `HeadTailLabel` → not selected.
8. Dive whose prediction names a *superseded* laser label → **selected**
   (mismatch, not absence).
9. Image with two valid laser labels → selected once, not duplicated.
10. `test_api_postgres_integration` — the selector against real Postgres over
    a deliberately multi-valued seed.

**predict_headtail_image (data-worker, 4-test structure)**
11. **Crop offset round-trips.** A keypoint at a known position in a crop taken
    at `(ox, oy)` comes back at that position plus the offset, in
    rectified-frame pixels. This is the one bug in the activity that would look
    entirely plausible in the output — every prediction quietly displaced by
    the crop origin — so it gets its own test rather than riding on the
    integration test.
12. Crop window clamping: a laser dot near any frame edge or corner still
    yields a full-size 1800×1350 window inside the frame, never a padded or
    truncated one.
13. Pure logic: laser inside instance 2 of a synthetic multi-instance mask set
    → picks 2; laser on background → `laser_off_all_fish`; nothing returned →
    `no_detections`; `find_head_tail_img` raising → `headtail_failed`.
14. Pure logic: `silhouette_ratio` = `mask_area_px / pred_len**2`, and the
    band check at the §0.5 boundaries.
15. In-process Temporal workflow contract test for the fan-out.
16. Integration (`-m integration`) against a real JPEG fixture, asserting the
    predicted length lands within tolerance of a known human label — the
    end-to-end check that the crop, the mask and the lift all agree.

**Populate**
17. Prediction present → task carries two `kp-1` keypoints labelled `Snout`
    and `Fork` at the right percentages.
18. Prediction absent → image is **not** populated (prediction-gated).
19. `rejected_low_confidence` → task seeded with **no** predictions, not
    skipped.
20. Re-run against a project that already has the task → no duplicate import.

**Tripwires**
21. `predict_headtail_image` imports no NAS client.
22. `test_schedule_registration` — +32 and +34 present, preprocess parent no
    longer dispatches populate.
23. The parent returns before staging anything when
    `ensure_gpu_worker_running_activity` reports `unavailable` (§1.2).

---

## 8. Rollout, and the thing that actually decides it

Ship dark first: run the predict parent and write `HeadTailPrediction` rows
for a week **without** the populate changes. Nothing reaches a labeler, and
the rows accumulate against labels that arrive independently — which
re-measures §0.2 on current data rather than on the 2026-09-02 backup.

Then enable seeding on **one dive**, and answer the question §0.6 says this
corpus cannot: does it help? The honest metric is labeler time per task and
the rate at which seeded points are moved, not agreement with the model.

**Kill switch.** `_RETIRED_SCHEDULE_IDS` + `retire_schedule` already exist and
actively delete a schedule at worker startup; that is how the slate detector
was shut down in a day. Seeding is one schedule (+34) and one gate, so
stopping it does not require unwinding the predictions.

The slate detector is the standing warning: it shipped behind an acceptance
gate calibrated on clear-water reef dives, and pool dives produced high-ECC
*false* fits that sailed through it. §0's numbers come from 151 dives across
the whole corpus, which is a much better base — but the failure mode to watch
is still a subpopulation the aggregate hides, which is why §0.5's gate is
recorded per row and retunable without re-predicting.

---

## 9. Open questions

1. **Does it save labeler time?** Unanswerable from this corpus (§0.6). §8 is
   the experiment.
2. **Is the fork error a definition problem or a model problem?** p50 7.4% vs
   2.8% for the snout. If labelers are systematically clicking the caudal fork
   while the detector returns a mask extremity, that is a fixable offset; if
   it is scattered, it is not. Answerable offline from the existing rows —
   worth doing before §8.
3. **~~Should the mask itself be seeded?~~ Superseded** — the multi-instance
   output is going to its own project instead. See
   [fish-detection-labeling.md](fish-detection-labeling.md). That work and this
   one are coupled in one direction: §0.2's 57.7% coverage is limited by the
   detector's *recall*, not by the laser gate, so a better detector lifts this
   stage's coverage directly. Nothing here blocks on it.
4. **Re-predict on supersede.** §2.1's `laser_label_id` makes the cohort
   drain, but nothing yet *invalidates* a prediction whose laser was
   superseded. Confirm the mismatch clause covers it, or add an explicit
   invalidation pass like `#527` did for stale measurements.
