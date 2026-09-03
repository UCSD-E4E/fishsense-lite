# Plan — a "Fish" Label Studio project: label every fish in the frame

Status: **proposed, not started.** Goal agreed 2026-09-02: **detection training
data first**, per-frame abundance counts as a by-product once recall is good
enough to trust.

A per-dive Label Studio project whose task is "box every fish in this frame",
seeded with `FishSegmentation`'s multi-instance output. Labelers correct, add
and remove; the corrected boxes become the training set for a better detector.

Companion to [headtail-prediction.md](headtail-prediction.md), and coupled to
it in one direction — see §0.4.

---

## 0. Evidence — measured 2026-09-02

Same corpus and tooling as the head/tail study
([`tools/validate_headtail_predictions.py`](../../tools/validate_headtail_predictions.py)),
re-read for the all-fish question. **The all-fish output has no ground truth in
the database** — the human labels only ever cover the one laser-designated fish
— so precision and recall here come from a hand audit of rendered overlays, not
from a metric. Treat §0.1 as directional, and see §9.1.

### 0.1 High precision, poor recall — and that is the good shape

Across 557 frames the segmenter returned **1,167 instances**, mean 2.10 per
frame:

| instances found | share of frames |
|---|---|
| 0 | 25.0% |
| 1 | 36.8% |
| 2–3 | 22.1% |
| 4–9 | 13.3% |
| 10–22 | 2.7% |

38% of frames carry two or more. On a stratified sample of frames rendered with
every instance outlined, inspected by eye:

* **Precision: no false positives on reef frames.** 16/16 instances across the
  inspected reef frames were real fish, with tight boundaries. (The false
  positives seen earlier — a diver's leg, a pair of swim shorts — were on pool
  fish-model dives, a distinct and unusual population.)
* **Recall: poor, everywhere.** Every inspected frame had obvious missed fish.
  One turbid pier frame containing dozens of schooling fish returned **zero**
  detections.
* **One instance-merging failure**: two overlapping fish returned as a single
  blob.

For model-assisted labeling this is the favourable asymmetry. Labeling work is
**additive** — labelers draw the missed fish rather than deleting garbage — and
false positives are the expensive error, because a seed that is often wrong
teaches labelers to distrust it. That is what went wrong with the slate
detector, and it is not the failure mode here.

### 0.2 A "zero detections" frame is not an empty frame

25% of frames get no detections, and the ones inspected were **full of fish**.
These are the model's blind spots, they are the highest-information frames in
the corpus, and a naive "seed what the model found" sampler would exclude
exactly them. §3 over-samples them deliberately.

### 0.3 Only 7.8% of the corpus has a rectified JPEG

Measured on a random sample of 400 canonical images against Garage:

| prefix | coverage |
|---|---|
| `preprocess_jpeg` (0.1) | 2.5% |
| `preprocess_headtail_jpeg` (5.1) | 6.2% |
| `preprocess_groups_jpeg` (2) | 6.2% |
| **any of the three** | **7.8%** |

So unlike the head/tail predictor — which reads a JPEG stage 5.1 already
wrote — **this project needs its own preprocess**: NAS stage + rawpy decode +
rectify + JPEG, at roughly 10 s/frame against 2 s for the inference. Preprocess
is the cost driver here, not the model, and that is what makes §3's sampling a
design problem rather than a formality. 65,981 canonical images across 272
dives, mean 554 per dive; exhaustive coverage is not on the table.

### 0.4 Detector + resolution benchmark (n=80, 71 dives)

Recall is bounded by **input resolution**, not by training data, and that is
measurable. `FishSegmentation` is Mask R-CNN (Fishial) with `MIN_SIZE_TEST=800`
/ `MAX_SIZE_TEST=1058`, so a 4014×3016 frame reaches it at **1058×795** — a
14.4× area reduction. Detection of the laser-designated fish tracks its size at
*model* resolution almost perfectly: **12%** below 75 px, **89%** above 198 px.

Scored against the laser oracle ("did any returned instance cover the validated
laser dot"):

| detector | mode | found the fish | excl. failures | s/frame |
|---|---|---|---|---|
| Fishial | full frame | 40/80 (50.0%) | — | 2.9 |
| Fishial | tiled 2200×1650 | 51/80 (63.8%) | — | 24.8 |
| SAM3 | full frame | 44/80 (55.0%) | 44/72 = 61.1% | **0.9** |
| SAM3 | tiled 1024, 0.2 | **58/80 (72.5%)** | **58/73 = 79.5%** | 11.3 |

SAM3's "failures" are CUDA OOM on a 6 GB laptop GPU (7–8 frames), counted as
misses; an artefact of the test machine, not the model.

Three things follow. **Tiling is worth more than the detector choice** and
costs no labels — it lifted Fishial by 13.8 points and SAM3 by 17.5, and lost
nothing (every fish the full frame found, tiling also found). **SAM3 dominates
Fishial on both axes** — better full-frame than Fishial *tiled*, at a third of
the time, and better tiled at half the time. And they are complementary: on the
tiled head-to-head SAM3 found 15 the other missed, Fishial 8.

SAM3 is not exempt from the ceiling — its encoder runs at `img_size=1008`,
essentially Fishial's 1058. It is better *per pixel of fish*, which is why it
still needs tiling.

Literature agrees this is not a data problem: FishDet-M (2025; 13 datasets, 28
models) reports AP_small 0.315 against AP_large 0.641 for its best model —
small-fish AP is half large-fish AP with the data problem already solved. SAHI
reports +5.1–6.8 AP from inference-only slicing with no retraining, and
+12.7–14.5 with slicing-aided fine-tuning, which is the arm labels would buy.

### 0.5 `FishSegmentation` was silently landscape-only — fixed upstream

Found while running the above: `inference` returned an all-zero mask for any
input where `width <= height`, with no error, ~40× faster than a real forward
pass. A square-tile sweep therefore scored **0/80** and read as "no fish
anywhere". Fixed in fishsense-core on `fix/segmentation-landscape-only`; until
that lands in a released wheel, **any tiling here must use landscape tiles**.

### 0.6 This fixes the head/tail stage's main limitation

The head/tail plan abstains on 42% of images: 25% no detection, 17%
"laser landed on no fish". Auditing the second category showed it is
predominantly **the detector missing the laser-designated fish**, not the laser
being off-target — dive 408 / image 116383 found 5 instances, all real fish,
and the laser'd fish was not among them.

So both abstention modes are recall failures, and better recall lifts head/tail
coverage directly. The coupling runs one way only; neither plan blocks on the
other.

---

## 1. What the project is

One LS project per dive, titled `"{dive.name} #{dive_id} - Fish Labeling"` via
`populate_utils.build_per_dive_title` — the `#{dive_id}` is not decoration, dive
names are not unique in prod and the create activity finds its project by title.

Labeling config, `FISH_LABELING_CONFIG_XML`:

```xml
<View>
  <RectangleLabels name="fish" toName="image">
    <Label value="Fish" background="#26a269"/>
  </RectangleLabels>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
</View>
```

One class, `Fish`. **Not species** — species labeling has its own project and a
taxonomy this must not duplicate; conflating them makes both tasks slower and
puts a second writer on `content_of_image`.

### 1.1 Boxes, not masks

The model outputs masks and masks would train a better segmenter, but a box is
what a labeler can correct in seconds and a mask is not — Label Studio's brush
tools are slow and imprecise at this image size. Boxes also match the existing
slate YOLO work, so the export path and training harness already have a
precedent.

Seed each box as the bounding box of its instance mask. The mask is discarded
at seed time but retained on the prediction row (§2.1) so a future segmentation
pass can use it without re-inferring.

---

## 2. Data model

### 2.1 `FishDetection` — what the model proposed

One row per (image, instance). Unlike `LaserPrediction` / `HeadTailPrediction`
this is not one row per image, so the natural key is
`(image_id, instance_index)`.

```python
class FishDetection(ModelBase, table=True):
    __table_args__ = (UniqueConstraint("image_id", "instance_index",
                                       name="uq_fish_detection_image_instance"),)
    id: int | None = Field(default=None, primary_key=True)
    image_id: int | None = Field(default=None, foreign_key="image.id")
    instance_index: int                      # 1..N within the frame
    x: float; y: float; width: float; height: float   # bbox, rectified pixels
    mask_area_px: int
    frame_width: int; frame_height: int      # for the LS percentage conversion
    created_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)
```

A frame the model found nothing in has **zero** rows, which is
indistinguishable from "not yet predicted". That distinction matters for the
cohort, so it is carried on the image-level row instead:

```python
class FishDetectionRun(ModelBase, table=True):
    __table_args__ = (UniqueConstraint("image_id", name="uq_fish_detection_run_image"),)
    image_id: int | None = Field(default=None, foreign_key="image.id")
    n_instances: int
    predicted_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)
```

Splitting the run from the detections is what makes §0.2's zero-detection
frames addressable as a population — they are the rows with `n_instances = 0`,
and they cannot be found any other way.

### 2.2 `FishLabel` — what the human drew

Mirrors the other four label kinds, one row per (image, project):

```python
class FishLabel(ModelBase, table=True):
    __table_args__ = (UniqueConstraint("image_id", "label_studio_project_id",
                                       name="uq_fish_image_project"),)
    id: int | None = Field(default=None, primary_key=True)
    label_studio_task_id: int | None = Field(default=None, unique=True, index=True)
    label_studio_project_id: int | None = Field(default=None, index=True)
    # [{x, y, width, height}, ...] in rectified pixels. A completed row with an
    # empty list is a real answer — "no fish in this frame" — and is the
    # negative example the training set needs most.
    boxes: List[Dict[str, Any]] | None = Field(default=None, sa_column=Column(JSON))
    label_studio_json: Dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    updated_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)
    superseded: bool | None = Field(default=False)
    completed: bool | None = Field(default=False)
    user_id: int | None = Field(default=None, foreign_key="user.id")
```

The empty-list case is called out because the obvious implementation drops it,
and hard negatives from frames that *look* like they should contain fish are
the most valuable rows in an object-detection training set.

Register both in `database.py` and the controller in `controllers/__init__.py`.

---

## 3. Frame sampling — the actual design problem

Preprocess is ~10 s/frame (§0.3) and labeling is human minutes, so the sample
is the budget. Two phases, because the stratification key does not exist until
after prediction:

**Phase A — candidate pool.** Per dive, take `3 × budget` canonical images
spread **evenly across the dive's timeline** (not the first N; a dive's opening
frames are setup shots and are not representative — the same correction the
validation sampler needed). Preprocess and predict those.

**Phase B — label set.** From the pool, pick `budget` frames stratified by
`FishDetectionRun.n_instances`:

| stratum | share of the label set | natural frequency |
|---|---|---|
| 0 instances | **1/3** | 25.0% |
| 1 instance | 1/3 | 36.8% |
| ≥2 instances | 1/3 | 38.2% |

The zero stratum is deliberately over-weighted relative to §0.1's natural 25%.
Those frames carry the most information and the least seeded help, so they are
also the slowest to label — budget accordingly.

Default `budget = 30` frames/dive, configurable. Across 272 dives that is
~8,160 labeled frames from ~24,500 preprocessed ones — about 68 CPU-hours of
preprocess, which at the hourly cadence and current replica counts is weeks,
not days. Start with a handful of dives (§8).

---

## 4. Workflows

```
PreprocessFishImagesParentWorkflow      (api-worker, hourly)
  select dive -> resolve Phase-A pool -> stage_raw -> child fan-out -> cleanup_raw
     child: preprocess_fish_image  (rectify -> fish_labeling_jpeg/{checksum}.JPG)

PredictFishImagesParentWorkflow         (api-worker, hourly)
  select dive -> resolve (images with a JPEG, no FishDetectionRun)
     -> child fan-out: predict_fish_image -> persist_fish_detections_activity

PopulateFishLabelStudioProjectParentWorkflow  (api-worker, hourly)
  create_fish_label_studio_project_activity(dive_id)
     -> Phase-B stratified pick -> import tasks with seeded RectangleLabels

SyncLabelStudioFishLabelsWorkflow       (api-worker, hourly)
  read completed tasks -> FishLabel rows (including empty-box rows)
```

The preprocess parent is a standard dispatch parent and should be built from
[`workflows/_dispatch.py`](../../services/fishsense-api-workflow-worker/src/fishsense_api_workflow_worker/workflows/_dispatch.py)'s
shared steps rather than copied from a sibling — that module exists precisely
because the six existing parents were copy-pasted and drifted.

`predict_fish_image` registers in the **`cpu`** role: `fishsense_core.fish` is
in the base wheel with ONNX statically linked and weights embedded in the
`.so`, ~2 s/frame on CPU, no checkpoint to bake into the image.

Schedule slots are contended between +30 and +55; propose **+22 preprocess-fish**
and **+24 predict-fish**, with populate and sync at +0 alongside the other
syncs (they select no dives). `test_schedule_registration.py` pins the stagger.

---

## 5. Export for training

`tools/export_fish_detection_dataset.py` — read-only, writes YOLO-format
labels + a manifest, splitting **by dive** (never by frame: consecutive frames
of the same fish would leak across the split). Includes completed `FishLabel`
rows with empty boxes as negatives. This mirrors what the slate YOLO work
already did, so the training harness is not new.

---

## 6. Test list — failing first

1. `FishDetection` upsert on `(image_id, instance_index)`; re-predict replaces
   rather than duplicating.
2. `FishDetectionRun` with `n_instances = 0` is distinguishable from no row.
3. `FishLabel` completed with `boxes = []` round-trips as a real answer, not
   NULL.
4. Phase-A pool is evenly spread, not head-of-list.
5. Phase-B stratification hits 1/3 each, and degrades gracefully when a dive
   has no zero-instance frames.
6. Pure logic: mask → bounding box, and the box → LS percentage conversion.
7. `predict_fish_image` returns N results for an N-instance label map, 0 for an
   empty one.
8. Workflow contract test for each fan-out.
9. Integration (`-m integration`) against a real ORF fixture.
10. Populate: seeded task carries one `RectangleLabels` region per detection.
11. Populate is idempotent — re-run imports no duplicate tasks.
12. Sync writes boxes in rectified pixels, converting from LS percentages.
13. Export splits by dive, and no dive appears in two splits.
14. `test_schedule_registration` — the two new slots.

---

## 7. What this does *not* do

* **No species.** One `Fish` class (§1).
* **No auto-accept.** Every box a labeler sees is a proposal; nothing writes a
  `FishLabel` without a human completing the task. This is the line the slate
  detector crossed.
* **No masks in Label Studio** (§1.1), though they are retained on the
  detection rows.
* **No counts yet.** Abundance is a by-product of good recall, and recall is
  the thing being fixed; publishing counts off the current detector would
  under-count badly and silently.

---

## 8. Rollout

**Do §0.4's tiling first, and pick the detector, before any labeling.** Both
are label-free, both move recall more than a corrected training set plausibly
would, and both change what the labeling budget should be. Concretely: switch
the seeding backend to SAM3 (it is faster *and* better in every cell of the
table), turn on tiling, re-measure coverage, and only then size §3.

Then start with **three dives**, chosen to span the conditions that matter: one
clear reef (where §0.1 says precision is good), one turbid pier dive (dive 108
is the standing example — a frame full of schooling fish returning zero
detections), and one pool dive (where the only observed false positives were).
Label them, measure how long a frame takes and how often a seeded box is
deleted, then decide the per-dive budget.

The deletion rate is the number that decides whether seeding helps at all. If
labelers routinely delete seeded boxes, the seeds are costing time rather than
saving it, and the project is better run unseeded until the detector improves.

---

## 9. Open questions

1. **Precision has no metric, only an audit** (§0). Before building, hand-label
   ~30 frames exhaustively and compute real precision/recall. That is a day of
   work and it converts the whole of §0.1 from directional to measured — and it
   doubles as the first evaluation set for the retrained detector.
2. **Instance merging** (§0.1) — one observed case of two overlapping fish in a
   single blob. Boxes make this cheap for a labeler to split, but it will show
   up as a systematic under-count in any abundance use.
3. **~~Does a corrected-box training set actually lift recall?~~ Answered —
   resolution first, labels second.** §0.4 settles it: the misses are small
   fish at model resolution, tiling recovers a large share of them with no
   labels at all, and the best of 28 contemporary detectors trained on 13
   datasets still halves its AP on small objects. So the sequence is **tile,
   re-measure, then set the labeling budget** — not the reverse. Labels are
   still worth buying afterwards (SAHI's fine-tuning arm roughly doubles the
   inference-only gain, and §9.1's evaluation set is needed regardless), but
   they are no longer the first lever.
4. **Where does the retrained model live?** fishsense-core owns
   `FishSegmentation`, and its weights are embedded in the wheel. A retrain
   means an upstream release and a wheel bump, and there is currently no path
   for this repo to carry its own detector weights the way it carries the laser
   detector's checkpoint.
