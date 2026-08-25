# Ingesting a dive

How to turn a folder of `.ORF` files on the NAS into a `Dive` and its `Image`
rows. Before this existed, rows were only ever created by an external crawler
that has since been retired and archived.

**One request means one dive.** The frames are the `.ORF` files *directly inside*
the folder you name — not a recursive walk. That is precedent, not a
simplification: the retired crawler assigned `dive = image.parent`, so every one
of the ~479 dive rows in production is exactly one directory. Recursing would
merge dives that are distinct rows today, silently and irreversibly.

---

## The short version

```bash
# 1. Dry run. Writes nothing, reports every problem at once.
temporal workflow start \
    --task-queue fishsense_api_queue \
    --type IngestDiveWorkflow \
    --workflow-id ingest-101923_Alligator1_FSL06-dry \
    --input '{"dive_path": "2024.06.20.REEF/101923_Alligator1_FSL06",
              "self_calibrates": true,
              "priority": "HIGH",
              "dry_run": true}'

temporal workflow result --workflow-id ingest-101923_Alligator1_FSL06-dry

# 2. Read the report. Fix anything in `errors`. Then commit:
temporal workflow start \
    --task-queue fishsense_api_queue \
    --type IngestDiveWorkflow \
    --workflow-id ingest-101923_Alligator1_FSL06 \
    --input '{"dive_path": "2024.06.20.REEF/101923_Alligator1_FSL06",
              "self_calibrates": true,
              "priority": "HIGH"}'

# 3. Watch it. A large dive is hours of downloading.
temporal workflow query \
    --workflow-id ingest-101923_Alligator1_FSL06 --type progress
```

**Always dry-run first.** It costs ~1 MB per frame instead of ~14.5 MB, because
preflight only reads each file's EXIF header. On a 500-frame dive that is 0.5 GB
rather than 7.5 GB, and it is the only chance to see every problem before
anything is written.

---

## The request

| Field | Meaning |
|---|---|
| `dive_path` | **Required.** Folder path relative to `e4e_nas.raw_root_path`. An absolute path also works and is not double-prefixed. |
| `self_calibrates` / `calibration_dive_id` | **Exactly one is required** — see below. |
| `priority` | `HIGH` (default) or `LOW`. HIGH is what every hourly cohort selects on. |
| `dive_name` | Defaults to the leaf directory name. |
| `camera_id` | Override camera resolution. Normally unset. |
| `dive_slate_id` | The dive's slate, if it has one. |
| `flip_dive_slate` | Passed through to the dive row. |
| `dry_run` | Stop after preflight, having written nothing. |
| `verify_existing` | **Declared but not implemented — it does nothing.** See "Known gaps". |

### Calibration intent is required, and cannot be inferred

Exactly one of `self_calibrates: true` or `calibration_dive_id: <id>`.

A fish-only dive with no slate frames of its own can never self-calibrate, so
stage 14 can never measure it — and *nothing in the files reveals that*. Ingest
refuses rather than guessing, because the failure mode is a dive that sits in the
pipeline forever without ever producing a measurement and without ever saying
why.

Passing both is also refused: `get_laser_extrinsics_for_dive` resolves own-wins,
so a dive with its own slate would silently ignore the link you took the trouble
to specify.

---

## What preflight refuses, and what to do

Preflight reports **every** problem at once rather than stopping at the first —
you get one round trip, not a sequence of them. A non-empty `errors` list means
nothing was written.

| Error | What it means | Fix |
|---|---|---|
| `Camera serial <S> matches no Camera row` | The frames' Olympus MakerNote serial has no `Camera`. | Add the camera **with its intrinsics**. Ingest deliberately does **not** fall back to the EXIF `Artist` tag — a free-text rig label would bind intrinsics belonging to different glass, and stage 14 would report confident wrong lengths. |
| `Frames span more than one camera serial` | One folder, two rigs. | Split the folder and submit each dive separately. The schema cannot express a two-rig dive. |
| `Camera <id> has no intrinsics` | The camera exists but has no calibration. | Add intrinsics. Otherwise the dive sits in the stage-14 cohort forever. |
| `No calibration intent given` / `Contradictory calibration intent` | See above. | Pass exactly one. |
| `No readable EXIF timestamp in <path>` | The frame has no usable tag 0x0132 or 0x9003. | Investigate the file. Ingest will not default a timestamp — stage-1 clustering is pure timestamp arithmetic and cannot tell a fabricated value from a real one. |
| `Path exceeds 255 characters` | `Image.path` is `varchar(255)`. | Shorten the path on the NAS. The offending file is named. |
| `No .ORF frames directly inside <folder>` | Usually a mistyped path — or you named a parent. | Check the path; see the subfolder warning below. |

### Warnings are informational — ingest proceeds

* **`<path> contains N .ORF files ... that is a separate dive`** — a subdirectory
  holding raws. Under the existing convention it is its own dive, so submit it as
  its own request. This is the Olympus rollover case: the TG-6 wraps its frame
  counter at `PA199999` and starts a child folder mid-dive, which reads as "more
  frames" to a person and as "a second dive" to every convention in the database.
* **`Dive <id> has the same folder name`** — a leaf-name collision. Dive names are
  not unique in production (dives 64 and 66 are both `082929_FishModels_FSL07`),
  so this is a hint, not a fault.
* **`EXIF Artist <A> disagrees with the resolved camera's name <N>`** — the serial
  is authoritative, so ingest proceeds. It usually means a mislabelled `Camera`
  row or a re-housed body, and nothing else in the pipeline would ever notice.
* **`<path> has no DateTime (0x0132); fell back to DateTimeOriginal`** — usable,
  but this body is not the one the convention was derived from.

---

## What actually happens on commit

Ingest is a **two-phase commit**, because no database transaction spans the steps.

1. **`create_dive`** writes the dive at **`priority=LOW`, whatever you asked
   for.** Every hourly cohort selects on HIGH, so a dive created HIGH before its
   frames land would be picked up mid-ingest and processed against a partial set
   — clustered on some of its frames, populated into Label Studio missing others.
2. **`scan_and_register_images`** downloads frames in batches of 25, computes each
   checksum, and creates the `Image` rows.
3. **`finalize_dive`** flips the dive to the priority you asked for — **and only
   if every listed frame is accounted for.**

**Priority is the commit flag.** A crash or a cancellation anywhere in the middle
leaves a dive and some images that no pipeline stage will touch.

`finalize` refuses, non-retryably, when:

* **any frame was rejected** — the set is incomplete;
* **`registered + skipped != total`** — a frame was neither written nor recognised
  as already present. That is worse than a rejection: a rejection is reported, a
  gap is silence.

Both leave the dive at LOW. Read the report, fix the cause, re-run.

### Re-running is safe

* `dives.post` upserts on `path`, so you get the same dive row, not a second one.
* The scan skips frames already registered **without downloading them**, so a
  re-run over a completed folder moves no bytes.
* A fully-skipped re-run still commits — otherwise the only way to re-verify a
  dive would be to make it fail.

---

## Duplicate content

After the scan, `finalize` reports **containment** against every other dive that
shares frames:

```
containment = |new ∩ existing| / |new|
```

over content checksums. `1.0` means this folder is wholly contained in that dive;
`0.87` means 48 of 55 frames already exist there. Being a set operation on hashes
it is immune to filenames and ordering, and it degrades to a partial overlap.

**It is reported, never blocking.** Re-ingesting the same frames under a second
dive path is legitimate and has already happened in production. The duplicate
rows land `is_canonical=False`, which makes them invisible to every pipeline
cohort — they cannot be measured twice.

About half of production's image rows are duplicate content, so expect this to
fire often.

---

## Verifying what is already there

Separate, read-only, and unrelated to a new ingest:

```bash
# One dive, sampling 25 frames
temporal workflow start --task-queue fishsense_api_queue \
    --type VerifyDiveChecksumsWorkflow \
    --workflow-id verify-checksums-412 --input 412 --input 25

# Every canonical dive, 5 frames each (~20 GB); pass null for every frame (~930 GB)
temporal workflow start --task-queue fishsense_api_queue \
    --type VerifyAllDivesChecksumsWorkflow \
    --workflow-id verify-sweep --input 5
```

It re-hashes real files and reports checksum mismatches, timestamp mismatches,
missing files and NULL checksums separately, because they have different
consequences. Run on 2026-08-17 across all 272 canonical dives: **1,619 frames,
zero checksum disagreements.**

---

## Known gaps

* **`verify_existing` does nothing.** The field exists on `IngestDiveRequest` and
  is honoured by no code. Use `VerifyDiveChecksumsWorkflow` above instead. The
  flag should be either wired up or removed — a flag that silently does nothing
  is worse than an absent one.
* **No API or portal route yet.** Ingest is CLI-only; the Temporal client and
  `ingest_controller` in fishsense-api are unbuilt (see
  `docs/plans/dive-image-ingestion.md` §2.7), and `/portal/ingest` after that.
