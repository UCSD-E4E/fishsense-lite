# Plan — laser calibration from a checkerboard

Stage 13 fits `LaserExtrinsics` from a **dive slate**. A large fraction of the
pool-test corpus was shot against a **checkerboard** instead, and those dives
cannot calibrate today — silently, and by design rather than by bug.

This is a plan for a second *producer* of `LaserExtrinsics`. It changes no
consumer.

---

## 0. What is verified, and what is not

### 0.1 VERIFIED — MOST calibration folders are checkerboards, but the corpus is MIXED

Surveyed all 15 laser-calibration folders across the three pool-test datasets
with `cv2.findChessboardCornersSBWithMeta` (+ `CALIB_CB_LARGER`, which reports
the grid it actually found), 6 sampled frames each:

| Folders | Detected | Board |
|---|---|---|
| 2025.01.17 ED-00, both rigs | 12/12 | **24 x 17 interior corners** (25 x 18 squares) |
| 2023.08.14 ED-00, all 8 | 46/48 | **14 x 10 interior corners** (15 x 11 squares) |
| 2023.08.18 rigs 05/06/07 | 18/18 | 14 x 10, same board |
| 2023.08.18 `04-slate`, `04-box-slate` | **0/12** | **not a checkerboard** |

**Rig 04 of 2023.08.18 used an actual dive slate** — a white cutting board with
a black hash pattern taped to it, hand-held, laser dot on it. That is one of the
`Tic-Tac-Toe` `DiveSlate` templates, so `04-slate` (dive 508) and
`04-box-slate` (510) can calibrate through the **existing** stage-13 path; they
only need `dive_slate_id` set by species labelling. They are not part of this
plan's problem.

An earlier revision of this section claimed all the folders were checkerboards.
That was an inference from two sampled frames, and it was wrong for 2 of 15
folders. The table above is measured.

The 2023 result independently confirms itself: the board is captioned
`15x11 checkerboard`, and detection returns 14 x 10 interior corners, which is
exactly what a 15 x 11 grid of squares has.

### 0.2 VERIFIED — nothing in the repo detects a checkerboard

`findChessboardCorners`, `charuco` and `aruco` appear nowhere in `services/`,
`libs/` or `tools/`. The only existing mention of a checkerboard is a **label**:
the species taxonomy already offers `Calibration Targets → E4E Checkerboard`
(added in #371, alongside `Ruler`). So the corpus can already be *marked*; it
just cannot be *used*.

### 0.3 RESOLVED — two boards, and only the E4E one is usable

The 2023 board is **14 x 10** interior corners; the 2025 board is **24 x 17**.

**Scope this to the E4E board (14 x 10) only.** Per the user, that is the one
they can use consistently, and OpenCV corner detection does not handle the other
board correctly. The survey agrees once you look past the detect/no-detect
count:

| Folder | Grids returned across 6 frames |
|---|---|
| 2025 FSL-10D | `(17,24)` — stable |
| 2025 FSL-11D | `(10,10) (15,24) (17,10) (17,24) (19,8)` — **five different grids in six frames** |
| every E4E folder | `(10,14)` / `(14,10)` — the same board, transposed |

A *wrong* grid is worse than a failed detection: it mis-pairs correspondences
and `solvePnP` accepts it silently (§4.6). An earlier revision of this section
reported the 2025 board as "12/12 detected", which counted those five
disagreeing results as successes. It is not usable.

The likely reason is resolution, not a defect in the board: 24 x 17 packs 408
corners into the same frame the E4E board fills with 140, so each square is far
smaller in pixels, and underwater blur and backscatter hit the fine grid hardest.

**Consequence: only one square size needs measuring** — the E4E board's — and
the 2025 pool-test dives (480-487) cannot be calibrated by this route at all.
480 and its borrowers are already parked (§0.5).

### 0.4 NOT VERIFIED — the square size, and this is the one that matters

Nobody has measured it. See §4.1: it is the single number that can silently
wreck every length this produces.

### 0.5 Why the affected dives fail silently

`perform_laser_calibration_activity` returns `None` (not an error) when
`dive.dive_slate_id is None`, and `Dive.dive_slate_id` is written *only* by
`sync_species_labels_...`, from the slate-type choices — which are exactly the
11 `DiveSlate` template rows. A checkerboard is none of them, so the best a
labeler can do is `SLATE_NOT_IN_LIST_LEAF`, which **deliberately** leaves
`dive_slate_id` NULL (see the taxonomy notes: a wrong slate is a wrong *scale*,
and that is the error reprojection cannot see).

The chain therefore stalls with no error anywhere:

```
no dive_slate_id -> stage 9 never fires -> no DiveSlateLabel rows
                 -> stage 13 returns None -> no LaserExtrinsics
                 -> every dive borrowing it is never measured
```

This is the *correct* failure direction — no wrong numbers are produced — but it
is invisible. The 2025 ED-00 dives (480-487) and the 2023 sets are all in it.

**Dive 480 and its borrowers 482 / 483 / 486 were parked at `Priority.NONE` on
2026-09-02** with notes recording why, its 28 laser labels superseded and its
Label Studio project deleted (the 4 annotations in it were synced to the DB
first — they are laser-dot positions on checkerboard frames, which is exactly
what this plan needs). Dive 481 and its borrowers 484 / 485 / 487 are in the
same position and have not been parked.

**The ingest intent is already right and needs no rework.** `self_calibrates:
true` means only "resolve from my own `LaserExtrinsics`"; under this plan dive
480 calibrates from its own checkerboard frames and 482/483/486 keep borrowing
it. Nothing needs re-ingesting or re-linking.

---

## 1. The seam: the slate only ever produces a *plane*

This is the finding the whole plan rests on. `_laser_point_in_camera_space`
does six things, and only the first three are slate-specific:

```python
# --- slate-specific -------------------------------------------------
body_points[:, :2] = (np.array(source_points) / slate.dpi) * INCH_TO_M   # 1
image_points       = label.reference_points                              # 2  (human)
ret, rvec, tvec    = cv2.solvePnP(body_points, image_points, K, zeros(5))# 3
# --- target-agnostic ------------------------------------------------
rotation, _        = cv2.Rodrigues(rvec)                                 # 4
slate_normal       = rotation[:, 2]
ray                = WorldPointHandler(K_inv).project_image_point(xy) * -1
scale              = (n @ p0) / (n @ ray)                                # 5
return ray * scale                                                       # 6  ray-plane hit
```

Steps 4-6 do not know what the target was. Neither does anything downstream:
`_gather_laser_points` -> `fishsense_core.laser.calibrate_laser` ->
`check_fit_self_consistency` -> `LaserExtrinsics` are all target-agnostic
already.

So the change is: **swap steps 1-3 for a checkerboard, keep 4-6 verbatim.**

Extract the shared half rather than copying it — `duplicate-code` will not catch
the clone (`DiveSlateLabel` vs a checkerboard DTO is a systematic rename, which
is textually invisible; see the CLAUDE.md note on exactly this shape):

```python
def plane_from_correspondences(body_points, image_points, K) -> tuple[normal, point] | None
def laser_point_on_plane(normal, point_on_plane, laser_xy, K) -> np.ndarray | None
```

Two callers produce `(body_points, image_points)`:

| Target | `body_points` | `image_points` |
|---|---|---|
| Dive slate | `reference_points / dpi * INCH_TO_M` | `DiveSlateLabel.reference_points` (human) |
| Checkerboard | `(col, row) * square_size_m` | `cv2.findChessboardCornersSB` + `cornerSubPix` (automatic) |

**`solvePnP` is called with `np.zeros((5,))` distortion.** That is correct only
because the pipeline feeds it rectified imagery (`RectifiedImage` applies
`cv2.undistort`). Checkerboard detection **must** run on the rectified image for
the same reason. Getting this wrong produces a plausible, slightly wrong pose.

---

## 2. Why this is a better calibration than the slate, not just a different one

Framed against the measured error budget, not in the abstract.

1. **It removes the error you cannot see.** Per the fish-model audit, scale error
   is the dominant term and is *invisible* to reprojection residual —
   `rho(residual_m, |pct_error|) = -0.026` over 1109 depths, and dive 84 had the
   worst residual with the best accuracy. Slate scale depends on picking the
   right one of 11 templates and trusting its PDF `dpi`; that is precisely where
   the 2026-08-11 panel-offset recalibration and the `V-Slate 7` sentinel came
   from. A checkerboard's scale is one caliper measurement of a repeating
   feature, and the board's grid is self-evident from the image.
2. **It needs no human in the loop.** The slate path runs species labeling ->
   stage 9 preprocess -> hand-placed `DiveSlateLabel` corners -> stage 13.
   `findChessboardCornersSB` is deterministic, and the laser dot already has a
   detector (`predict_laser_image`, `LaserDetector`, GPU queue) plus RANSAC
   outlier validation.
3. **It is enormously better conditioned.** A 15x11 board gives 154 interior
   corner correspondences per frame at subpixel precision, versus a handful of
   hand-clicked slate points. `solvePnP` stops being the weak link.
4. **There is far more data than the fit needs.** `MIN_LASER_POINTS = 2`, and
   these folders hold 28-133 frames each. That is enough to fit robustly, report
   spread, and reject outlier observations instead of hoping two were good.

### 2.1 A simplification worth knowing before scoping

**Only the plane matters, not the pose.** Steps 4-6 use `rotation[:, 2]` (the
normal) and one point on the board. Nothing reads the in-plane rotation or the
origin corner. So the classic checkerboard headaches — 180° ambiguity, which
corner is (0,0), consistent corner ordering between frames — are all irrelevant
here. A checkerboard is *easier* for this job than for the extrinsics job it is
normally used for.

---

## 3. Data model — two options

`LaserExtrinsics` itself needs **no change**: one row per dive
(`uq_laserextrinsics_dive_id`), `laser_position` (z padded to 0 by the Rust
kernel), `laser_axis`, `camera_id`. Stage 14, laser depth, the `calibrated`
flag and `Dive.calibration_dive_id` borrowing all keep working untouched.

### Option A — a real `CalibrationTarget` (recommended)

New table: `id`, `name`, `kind` (`slate` | `checkerboard`), `rows`, `cols`,
`square_size_m`, `created_at`. New nullable `Dive.calibration_target_id`.

* Honest naming, room for ChArUco later, and `square_size_m` is a first-class
  measured quantity rather than an inferred `dpi`.
* Costs a migration, an SDK mirror (+ `test_sdk_drift` entry), a cohort
  predicate, and a way to set the link.

### Option B — reuse `DiveSlate`, zero schema change

A checkerboard *is* a planar grid of known points, and `DiveSlate` already
stores `reference_points` + `dpi`. Insert a row named `E4E Checkerboard 15x11`
whose `reference_points` are the 154 interior corners; write `DiveSlateLabel`
rows automatically from detection. **Stage 13 then needs no change at all.**

* Cheapest possible path, and it reuses a well-tested code path.
* But it calls a checkerboard a "dive slate" in the schema, in Label Studio, and
  in `dive_pipeline_status.slate_*`. That is a name that will mislead someone
  later, and this codebase has been bitten by exactly that (the four
  `put_*_label` clones, the drifted stage-13 comments).

**Recommendation: A**, with §1's seam extracted first. The seam is the valuable
part and is identical either way; do it as its own pure-refactor PR (no
behaviour change, existing stage-13 tests must stay green) so the new producer
lands on top of something already proven.

---

## 4. Risks

### 4.1 The square size is the whole ballgame — measure it, do not trust the print

This is the ruler lesson repeating. The `fishmodelreference` Ruler was assumed
to be 14 in / 355.6 mm; the true clicked span was **342.9 mm**, a 3.3% scale
error that was tick-verified twice before anyone believed it. A printed
checkerboard carries the same risk from print scaling, paper shrinkage and
lamination.

A nominal "15x11 @ 25 mm" from a PDF is **not** evidence. Measure a span of many
squares with calipers and divide — measuring one square multiplies your reading
error by the grid count. Record it per physical board, and treat a board that
cannot be re-measured the way `V-Slate 7` is treated: refuse to calibrate rather
than guess.

### 4.2 Only the E4E board is in scope

See §0.3. One board, one square size to measure. If the 2025 board is ever
wanted, it needs a detector that copes with its pitch — ChArUco, or a
fine-grid-tuned `findChessboardCornersSB` — not the current one.

### 4.3 Refraction and the depth regime

Intrinsics come from underwater checkerboards (`LensCalibration`), so the flat
port is partly absorbed into `CameraIntrinsics`. But depth-dependent residual is
already measured — the ruler read -2% at 0.9 m and -5% at 2.1 m. A calibration
shot at 0.5 m and applied to fish at 3 m sits in that gap. Worth measuring
across the board's own depth range, which these folders happen to span.

### 4.4 Dot detectability varies with what it lands on

In the 2025 sample the green dot sits on a white square; in the 2023 sample the
red dot straddles a black/white boundary. Contrast against a black square is a
different detection problem, and the existing `LaserDetector` was trained on
reef/pool scenes, not on a checkerboard. Expect to validate or fine-tune it.

### 4.5 Partial detection is common — and is fine, if handled

`CALIB_CB_LARGER` returned a **sub-grid** on many frames (6 x 14, 14 x 8,
17 x 10, 19 x 8 ...), where occlusion, glare or the frame edge cut the board
down. Detection rate is nonetheless ~96% overall, and `MIN_LASER_POINTS = 2`
against folders of 28-133 frames means the fit is heavily over-determined
either way.

A partial detection is **usable**, and for a reason worth stating: a sub-grid of
a regular grid has the same pitch, and by §2.1 only the plane matters — so the
unknown offset of the sub-board into the full board never reaches the answer.

The consequence for the producer is concrete: build `body_points` to match the
**detected** grid shape, not the nominal board. Generating a full 24 x 17 body
grid against a 17 x 10 detection would mis-pair the correspondences, which
`solvePnP` accepts silently (§4.6).

Prefer `findChessboardCornersSB` (robust, subpixel built in). ChArUco would make
partial views unambiguous and is the natural upgrade if this proves fiddly.

### 4.6 Detection must run on rectified, full-resolution frames

Two separate constraints:

* **Rectified**, because `solvePnP` is given zero distortion (§1).
* **Full resolution** — at half scale the 2023 board dropped from 14 to 13
  detected columns. A silently smaller grid is a mis-pairing risk, not just
  fewer points.

The survey above ran on camera JPEG siblings rather than rectified frames, so
its corner *positions* are distorted and unusable for a pose. Grid size and
detect/no-detect are unaffected, which is all it was for.

---

## 5. Recompute is free

Both `LaserDepth` (`laser_label_id` + `laser_extrinsics_id`) and `Measurement`
(`laser_extrinsics_id`) record which calibration produced them, and both cohorts
select on **mismatch** rather than absence. So a dive that gains — or improves —
its `LaserExtrinsics` re-enters the depth and measurement cohorts on its own and
drains at the hourly cadence. No backfill script is needed; this is the property
added on 2026-08-18 for exactly this situation.

---

## 5.1 What this actually covers

E4E-board calibration dives, all on the 2023 sets:

* **2023.08.14** — 488, 489, 490 (FSL-02D x3), 493, 496, 499, 502, 505
* **2023.08.18** — 512, 515, 518 (rigs 05/06/07)

Eleven calibration dives, and through `calibration_dive_id` they carry the
`Box` / `George` dives of those same rigs. Rig 04 of 2023.08.18 (508, 510) is a
real dive slate and uses the existing path. The 2025 dives (480-487) are out of
scope.

## 6. Suggested order

1. **Pure refactor.** Extract `plane_from_correspondences` +
   `laser_point_on_plane` from `_laser_point_in_camera_space`. No behaviour
   change; existing stage-13 tests are the gate.
2. **Measure the boards.** Calipers, per physical board, recorded. Blocks
   everything real (§4.1).
3. **Detection kernel, offline.** `findChessboardCornersSB` on rectified frames
   from these dives; report detection rate per folder. Read-only, and it answers
   §0.3 and §4.5 with data rather than a sample of two.
4. **Target model + cohort** (Option A), then the sibling activity, then wire it
   into `PerformLaserCalibrationParentWorkflow` beside the slate path.
5. **Validate against a known answer before trusting it.** The 2023 sets include
   rigs (`FSL-02` .. `FSL-07`) that already have slate-derived calibrations and
   measurements in prod. Fit those dives both ways and compare, and check the
   fish-model lengths against `fishmodelreference` at p90 — never an uncentered
   mean (foreshortening is one-sided).

## 7. Open questions

* Square size of the **E4E board** — 14 x 10 interior corners, 15 x 11
  squares (§4.1). One measurement now, not two. **Blocking.**
* ~~Are all `LaserCalibration` folders checkerboards?~~ **Answered** (§0.1):
  13 of 15 are; rig 04 of 2023.08.18 used a real dive slate.
* ~~Is the 2025 board a different geometry?~~ **Answered** (§0.3): 24 x 17
  versus the E4E board's 14 x 10 — and it is out of scope, because OpenCV does
  not detect it reliably.
* Should `Calibration Targets → E4E Checkerboard` in the species taxonomy become
  the identification hook, the way `'Slate, Laser on slate'` gates stage 9? It
  already exists, is already labelable, and now names exactly the one board in
  scope — but it would put a human back in a loop this design can otherwise run
  without.
