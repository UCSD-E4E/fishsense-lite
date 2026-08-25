# Plan — dive/image ingestion + the ungradeable `Weasly Fish`

Status: **approved; in progress.** Two independent deliverables:

* **Part 1** — ingest: API write endpoints, an api-worker Temporal workflow that
  scans a NAS folder, and a `/portal/ingest` page.
* **Part 2** — `Weasly Fish` reference row (small, ships independently).

**Progress**

| PR | Scope | State |
|---|---|---|
| 1 | fishsense-api write endpoints + SDK client methods (§2.1, §2.2, §4.2.1 lookup) | **merged** — #554 |
| 1b | canonical-only pipeline work: 11 cohort selectors, 8 resolvers, `dive_pipeline_status` | **merged** — #555, folded into #554's branch |
| 2 | api-worker ingest workflow | **in progress** — see below |
| 3 | `/portal/ingest` page (§5) | not started |
| 4 | Part 2 — `Weasly Fish` reference row (§7) | blocked on the calipered widths |

**PR 2 breakdown** (branch `feat/ingest-workflow`):

| Piece | State |
|---|---|
| `exif.py` — stdlib ORF reader, tag 0x0132 + Olympus MakerNote serial | **merged** — #557 |
| `NasClient.list_dir` + `download_range` | **merged** — #557 |
| `fishsense_shared.ingest_contracts` | **merged** — #557 |
| `list_dive_folder_activity` (§4.1) | **merged** — #559, 11 tests |
| `preflight_ingest_activity` (§4.2) | **merged** — #559, 19 tests |
| `create_dive` / `scan_and_register_images` / `finalize_dive` (§4.3–4.5) | not started |
| `IngestDiveWorkflow` + `progress` query | not started |
| Temporal client + `ingest_controller` in fishsense-api (§2.7) | not started |

Unplanned, added while building the above because §0.1's caveat turned out to be
answerable rather than merely noted:

| Piece | State |
|---|---|
| `verify_dive_checksums_activity` + `VerifyDiveChecksumsWorkflow` (§4.6) | **merged** — #559 |
| `VerifyAllDivesChecksumsWorkflow` — the corpus-wide sweep | **merged** — #567 |

Read §0 first: one required verification is **blocked in this session**, and one
finding **contradicts an assumption in the brief**.

---

## 0. What I verified, what I couldn't, and what contradicts the brief

### 0.1 The checksum: **RESOLVED from source** — plain MD5 of the whole file

The legacy repos you pointed me at settle it. The ingest tool is
`UCSD-E4E/fishsense-data-processing-spider`, and
`fishsense_data_processing_spider/backend.py:67`:

```python
def get_file_checksum(path: Path) -> str:
    cksum = md5()
    with open(path, 'rb') as handle:
        for blob in iter(lambda: handle.read(8192), b''):
            cksum.update(blob)
    return cksum.hexdigest()
```

Called from `discovery.py:272` and written straight into `images.image_md5`
(`sql/insert_image_path_dive_md5.sql`), which is the column `9e5bc64`'s migration
copied into `Image.checksum`.

**So `Image.checksum == hashlib.md5(<entire file bytes>).hexdigest()`.** The 8192-byte
streaming is chunking, not scoping — byte-identical to hashing the whole buffer at
once. `git log -S` over `backend.py` shows the function introduced once and never
modified. No prefix variant, no canonicalisation, no header skipping. The prior
plan's blocking gate is **dropped**.

Two caveats worth keeping:

* This proves what *spider* wrote. It did not prove every row came from spider —
  older rows could predate it — so §4.6 stayed in the plan as a "cheap standing
  net". **It has now been run, and the answer is clean.**

  `VerifyAllDivesChecksumsWorkflow` re-hashed real files off the NAS on
  2026-08-17, across **all 272 canonical dives**, twice:

  | Sample | Frames | Checksums matched | Dives errored |
  |---|---|---|---|
  | 1 frame/dive | 272 | **272 / 272** | 0 |
  | 5 frames/dive | 1,347 | **1,347 / 1,347** | 0 |

  So `Image.checksum` really is `md5` of the whole file, corpus-wide. This is
  what the §4.2.1 containment design rests on: had the migration hashed
  differently, duplicate detection would not have errored — it would have
  silently reported *zero overlap*, every re-ingested frame would have landed
  `is_canonical=True`, and #555's gating would have had nothing to gate on.

  **Timestamps are a different story — two dives disagree**, and the two
  disagree differently:

  * **Dive 430** — one frame of five, `PA150459.ORF`: stored `11:08:00Z`,
    actual `11:08:58Z`. Seconds truncated. Sporadic, harmless to stage-1
    clustering (clusters are minutes apart).
  * **Dive 294** — *all five* sampled frames: stored `2023-11-20T00:00:00Z`
    (midnight, date-only) and EXIF **unreadable**. The filenames are
    `P1010001.ORF`… — not the Olympus `PA…`/`PB…` pattern, so probably another
    body whose MakerNote neither our reader nor the migration could parse.
    Dive-wide, and **not** harmless: every frame collapses to one instant, and
    stage-1 clustering is pure timestamp arithmetic.

  Note what preflight does with dive 294's case: it **rejects** a frame with no
  readable timestamp rather than defaulting one (§4.2). That is the intended
  behaviour, now confirmed against a real file the migration guessed at.
* Ingest streams from the NAS anyway, so implement it as the same 8192-byte
  streaming update rather than `md5(f.read())` — a 15 MB buffer per file times a
  batch is real memory for no benefit.

### 0.2 Contradiction: `NasClient` can't list, but the underlying client can

The brief says a listing method "must be added". True of
[nas.py](services/fishsense-api-workflow-worker/src/fishsense_api_workflow_worker/nas.py),
but the wheel it wraps already has it — no new dependency, no vendoring:

```
>>> [m for m in dir(synology_filestation._native.Client) if not m.startswith('_')]
['create_folder', 'delete', 'download', 'download_to', 'exists', 'getinfo',
 'list_dir', 'list_shares', 'login', 'logout', 'upload', 'upload_bytes']
```

`list_dir(path)` returns dicts with `path` / `isdir` / `size` / `mtime`
(shape confirmed from the wheel's own `fsspec.py` adapter). So `NasClient.list_dir`
is a ~10-line wrapper in the existing style, matching `download_to` / `exists`.

**Bonus that shapes the design:** `Client.download(path, *, offset=0, length=0)`
does **ranged** reads. EXIF lives in the first ~1 MB of an ORF, so the dry-run
pass reads ~1 MB/file instead of ~15 MB — a 500-image dive previews for ~0.5 GB
instead of 7.5 GB. This is what makes the dry-run affordable (§3.2).

### 0.3 **CONFIRMED**: `taken_datetime` is camera wall clock labelled UTC — and it is tag 0x0132, not 0x9003

I had this from circumstantial evidence; the spider confirms it and corrects the
tag I picked. `backend.py:20`:

```python
img = Image.open(path)
exif = img.getexif()
creation_time_str = exif.get(ExifTags.Base.DateTime)          # tag 0x0132
creation_time = dt.datetime.strptime(creation_time_str, '%Y:%m:%d %H:%M:%S')
return creation_time                                           # NAIVE — no tzinfo
```

Three corrections to what I had written:

1. **The tag is `DateTime` (0x0132, IFD0), not `DateTimeOriginal` (0x9003).** They
   are equal on the fixture and usually equal in practice, but they are not the same
   field — 0x0132 is the camera's file-modify stamp. Ingest must read **0x0132** to
   match, with 0x9003 as a fallback only when 0x0132 is absent (and log when it falls
   back).
2. **The datetime is naive.** Written to legacy `images.date TIMESTAMP` (no tz —
   `2025-03-15_update_table.sql`), then copied by `9e5bc64` into
   `Image.taken_datetime TIMESTAMPTZ`, picking up UTC. The chain is now fully
   traced end to end. `OffsetTimeOriginal` was never read; the camera-local offset
   was discarded at the source.
3. My circumstantial reading (hours 6–9, `TzInfo(0)`, Florida in August) was right.
   Nothing further to confirm against prod.

**Decision unchanged:** reproduce it. Parse 0x0132, attach `timezone.utc`. Capture
`OffsetTimeOriginal` into the report for the record and WARN when it is present and
non-zero, but do not apply it. Correcting the column is a backfill decision across
~111k rows, not ingest's call.

**Also: don't use Pillow.** Spider pinned Pillow 11.3.0; on current Pillow (12.3.0,
what we'd ship) `Image.open` on the real ORF fixture raises
`cannot identify image file` — ORF's magic is `IIRO`, which `TiffImagePlugin`
rejects. The stdlib reader in §0.7 has no such problem and I have run it against
the real file. Reading 0x0132 instead of 0x9003 is a one-line change to it.

### 0.4 Historical semantics, now traced through both repos

| Field | Rule, and where it comes from | This plan |
|---|---|---|
| `Image.is_canonical` | `9e5bc64`: `existing_checksum is None` — first row for a checksum wins. Replaces spider's `canonical_dives` table, which keyed on a whole-dive checksum (`MD5(STRING_AGG(basename \|\| ':' \|\| image_md5))`, lowest path wins). | **Reproduce**, computed **server-side in the POST** (§2.2). A client-side check races itself. This is the dives-64/66 case. |
| `Dive.dive_datetime` | Two different things existed: spider set `dives.date` = **mean** of image dates truncated to a DATE; `9e5bc64` **ignored that** and used `sorted(image_dates)[-1]` — the **max**. The max is what is in the column today. | **Reproduce the max.** |
| `Dive.camera_id` | `9e5bc64`: `images[-1]["camera_sn"]` — the last image *globally*, a real bug; every dive got whatever camera the final image of the whole dump used. | **Diverge**: resolve per dive from that dive's own frames. Existing values are unreliable — audit separately, §9. |
| multi-camera dives | Spider explicitly refused to assign a camera when a dive's frames spanned >1 serial, and wrote them to a `multiple_camera_dives` report file. | **Adopt as a hard preflight failure** — a mixed-rig folder is an operator error, and the wrong intrinsics silently corrupt stage 14. |
| dive identity | Spider: `dive = image.parent.relative_to(data_root)` — see §4.1, this changes the workflow's shape. | **Reproduce.** |
| dropped columns | `invalid_image`, `multiple_date`, `ignore`, `laser_task_id`, the whole `canonical_dives` table. | Not reintroduced. |

### 0.5 Camera identification: **serial number** (your call), and it needs no exiftool

Spider keyed on the Olympus **MakerNote** serial (`backend.py:136`,
`et.get_tags(paths, ['MakerNotes:SerialNumber'])`), which is what populated
`Camera.serial_number`. **Decision: key on the serial.** It is the safer key and the
one your existing rows already use — `Camera.name` is an operator-settable label and
EXIF `Artist` can be re-typed on the camera body; a body serial cannot.

My earlier "use `Artist`, MakerNote parsing is too fragile" recommendation rested on
a broken parse, and it was wrong. Two bugs in my first attempt:

1. Olympus sub-IFD offsets are relative to the **start of the MakerNote block**, not
   to the file.
2. The Equipment pointer (tag `0x2010`) has **type 13** — a non-standard IFD type
   Olympus uses — which my type-size table didn't know, so the entry was silently
   skipped. That is why the parse "returned garbage": it never reached the sub-IFD.

With both fixed, ~70 lines of stdlib extract it from the real fixture:

```
Olympus2 MakerNote @ 3092, header b'OLYMPUS\x00II\x03\x00'
  tag=0x2010 typ=13  -> Equipment sub-IFD @ base+114
    tag=0x0101 SerialNumber -> 'BJ6C67989'
```

`BJ6C67989` matches the serial visible in the file's raw bytes. **No exiftool, no
Perl runtime in the api-worker image, no subprocess per batch.**

Resolution order:

1. Explicit `camera_id` on the request — always wins.
2. MakerNote `SerialNumber` → `Camera.serial_number`.
3. **No fallback.** Unreadable serial, or one matching no `Camera` row → preflight
   **fails loudly**. Silently falling back to `Artist` is precisely the bug class
   that yields a dive with wrong intrinsics and no error anywhere.

`Artist` is still parsed, but only as a **cross-check**: if it disagrees with the
`Camera.name` the serial resolved to, preflight warns. That catches a mislabelled
`Camera` row, which nothing else would.

Caveat stated plainly: verified against **one** TG-6 fixture. The Olympus2 format is
shared across the TG line, but a non-Olympus body would return `None` — which fails
preflight rather than guessing. The `camera_id` override is the escape hatch.

**Lookup mechanics:** the `camera` table is tiny (a handful of rigs), so the activity
does one `cameras.get()` and builds a `{serial_number: id}` map in memory rather than
adding a `GET /cameras/serial/{sn}` route. Cheaper than N lookups, one less route.

**Gap this exposes:** there is no endpoint to *create* a `Camera`. A genuinely new
rig therefore cannot be ingested until someone inserts the row (DB-direct today).
Deliberately out of scope — noted in §9.

### 0.9 Prod state, measured 2026-08-07 — the canonical invariant holds

Checked directly against the prod `fishsense` DB before shipping the
`uq_image_canonical_checksum` partial unique index, because a unique index that
fails on existing rows would crash-loop the API (`lifespan` runs
`run_alembic_upgrade` on startup, and there is no staging tier):

```
total_images                      = 131,430
canonical_rows                    =  65,981
distinct_checksums                =  65,981
checksums_with_multiple_canonical =       0
```

`canonical_rows == distinct_checksums` with zero multi-canonical groups means
every distinct checksum has **exactly one** canonical row. So the migration
takes the create path, not the `REFUSING` path, and no operator repair is
needed. (The same arithmetic rules out checksums with *zero* canonical rows —
the counts would differ — and any NULL `is_canonical` could only sit among
duplicates, which a `WHERE is_canonical` partial index does not constrain.)

**49.8 % of image rows are duplicate content** (131,430 total vs 65,981
distinct). That reframes two things:

* `is_canonical` is not an edge case — it decides the identity of half the
  table. The promotion-must-demote bug the constraint caught was live on a much
  bigger surface than the dives-64/66 anecdote implied.
* Duplicate detection (§4.2.1) will fire constantly, not rarely. It also
  explains why the legacy whole-dive MD5 aggregate disappointed: an
  all-or-nothing digest over a corpus that is half near-duplicates is wrong
  most of the times it is consulted.

### 0.6 Security call-out — resolved

The `fabricant-prod` Postgres password committed in plaintext (this repo's `9e5bc64`
migration notebook, and cells in the spider repo) **has already been rotated** —
confirmed 2026-08-06. Recorded so nobody re-reports it. Nothing to do.

### 0.7 EXIF reader: stdlib, no new dependency

`Pillow` cannot open ORF (`cannot identify image file`). `rawpy` is a data-worker
dep and deliberately absent from the api-worker. Rather than add `exifread`, I
wrote and **ran** a ~60-line stdlib TIFF/IFD reader against the real fixture — the
output in §0.5 is its actual output. ORF is TIFF with magic `IIRO`; parsing IFD0
+ the Exif sub-IFD (0x8769) for tags 0x9003/0x9011/0x010F/0x0110 is all we need.

Ships as `fishsense_api_workflow_worker/exif.py`, unit-tested against a small
committed header fixture. Zero deps, and it works on ranged first-1 MB reads, which
the dry run needs.

The same module carries the Olympus MakerNote walk for `SerialNumber` (§0.5) —
type-13 IFD pointers and base-relative offsets included, both verified against the
real fixture.

---

## 1. Design decisions (direct answers)

**Where the scan runs — Temporal workflow on `fishsense_api_queue`. Confirmed.**
The api-worker is the only service with NAS creds and that is policy, not
accident. Volume settles it: 500 × ~15 MB ≈ 7.5 GB at
`e4e_nas.stage_concurrency` = 1 (deliberately serial — FileStation's download
backend 502s under concurrency, krg-infra#501). Tens of minutes to hours per
dive. The HTTP endpoint starts the workflow and reports; it never scans.

**Idempotency.** Two natural keys, both unique, both upserted the resolve-then-merge
way that #347 forced (`post_measurement` is the template):

* `POST /api/v1/dives/` — if `id is None`, `SELECT Dive WHERE path = :path`; adopt
  the id if found; then `session.merge`.
* `POST /api/v1/dives/{dive_id}/images/` — same against `Image.path`.

A blind `session.merge(..., id=None)` INSERTs and blows up on the unique index on
the second run. Both endpoints get an explicit "run it twice" test (§6).

`checksum` is deliberately **not** an idempotency key — duplicate content across
dives is expected and legitimate (dives 64/66). It only drives `is_canonical`.

**`taken_datetime`.** EXIF **`DateTime` (0x0132)** from IFD0 — the tag spider
actually read (§0.3) — with `DateTimeOriginal` (0x9003) as a logged fallback when
0x0132 is absent. `tzinfo=UTC` attached to the naive value, reproducing the
existing convention. Missing or unparseable → **the image is rejected, not
defaulted**. It lands in `IngestReport.rejected` with a reason, the workflow keeps
going, and the final activity **refuses to flip the dive to HIGH if anything was
rejected**. A fabricated timestamp would silently corrupt stage-1 clustering,
which is pure timestamp math; a missing image is visible and fixable.

Spider was weaker here: `get_image_date` returned the exception, the image simply
never got a date, and the dive proceeded with `NULL` — `query_dates_from_dive`
just skipped it. That silent-skip is the behaviour this plan deliberately does
not reproduce.

**Partial failure at image 300 of 500.** Three layers:

1. *The dive is created at `priority=LOW`.* No cohort selector looks at LOW dives,
   so a half-ingested dive cannot enter the pipeline. **`priority=HIGH` is the
   commit flag**, set by the final activity only when every listed file is
   persisted and nothing was rejected. This is the single most important safety
   property in the design — it is what makes a crash inert rather than corrupting.
2. *Batches of 25.* Images are scanned and posted in batches, one activity per
   batch, with `activity.heartbeat(i)` per file. A crash loses at most the current
   batch; heartbeat details let a retry resume mid-batch. Files 1–299 are already
   persisted rows.
3. *Re-run is resume.* Re-running with the same `dive_path` upserts the dive by
   path, skips images whose path already exists (no download at all — the skip is
   decided before the NAS call), and resumes at 300.

**How the portal talks to Temporal.** The web app has no Temporal client and must
not get one (that would need mTLS certs in a public-facing container). Path:

```
browser  ──"use server" action──▶  Next.js server (re-checks auth())
         ──Basic auth fetch────▶  fishsense-api  POST /api/v1/ingest/dives
                                     └─▶ temporalio Client.start_workflow(
                                            "IngestDiveWorkflow", …,
                                            task_queue="fishsense_api_queue",
                                            id=f"ingest-dive-{slug}")
                                  ◀── {workflow_id, run_id}
poll 3s  ──action──▶ GET /api/v1/ingest/dives/{workflow_id}
                                     └─▶ handle.describe()  (status)
                                     └─▶ handle.query("progress")  (counts)
```

**Decision: the Temporal client goes in fishsense-api** (your call, 2026-08-06).
Concretely (details in §2.7): a `temporalio` dep, an **optional** `[temporal]` config
block, the krg-prod cert mount on the api container, and a workflow-type allowlist so
the API can start exactly one workflow type and nothing else.

The property that must hold: **the API still boots and serves every existing route
when Temporal is unreachable or unconfigured.** Ingest is a new capability bolted
onto a service carrying a lot of load-bearing read traffic; it must not become a new
way for that service to fail to start. §2.7 is mostly about that.

*Alternative, now dropped:* an `IngestRequest` table polled by a new schedule. It
would have avoided the dep and the cert mount, at the cost of a table, a schedule
slot, a state machine, up-to-N-minutes latency on a button the operator is watching,
and a status read-back path it still needed. Recorded so the trade-off stays legible
if this is ever revisited.

**Validation that fails loudly.** All four, in a `preflight_ingest_activity` that
runs **before any write** and raises a non-retryable `ApplicationError` carrying
every failure at once (not first-wins — an operator should see all the problems in
one round trip):

| Check | Failure mode it prevents |
|---|---|
| camera resolves (override or `Artist`→`Camera.name`) **and** `GET /cameras/{id}/intrinsics/` returns 200 | stage 14 cannot measure; nothing errors, dives just never become `measured` |
| `priority == HIGH` in the request | no cohort selector ever picks the dive up; it sits invisible forever |
| exactly one of `calibration_dive_id` / `self_calibrates=True` given; if `calibration_dive_id`, that dive must exist | a fish-only dive with no slate frames can never be calibrated — **and this cannot be detected from the files**, so the operator must state intent at request time |
| every `Image.path` and the `Dive.path` ≤ 255 chars | silent DB truncation / insert failure mid-run |
| ≥1 `.ORF` found; dive path not already owned by a different dive id | empty or mis-targeted ingest |

The third one deserves emphasis: it is the only one that is *unknowable* from the
data. Making it a required field is the only way to make it loud.

---

## 2. API surface

### 2.1 `POST /api/v1/dives/` → `int`

Body is a `Dive`. Upserts on `path`. Validates: `camera_id` set, exists, and has a
`CameraIntrinsics` row (else 422); `calibration_dive_id`, if set, exists and
`!= id` (else 422); `len(path) <= 255` (else 422). Returns the dive id.

Also serves **finalize** — the workflow POSTs the same path again with
`dive_datetime` / `name` / `priority=HIGH` filled in.

### 2.2 `POST /api/v1/dives/{dive_id}/images/` → `int`

Body is an `Image`. Forces `dive_id` from the path. Upserts on `Image.path`.

**`is_canonical` is computed server-side when the body omits it**: `True` iff no
other `Image` row already carries this `checksum`. This is `9e5bc64`'s rule moved
to where it belongs — a client-side check races itself and gets the answer wrong
under any concurrency. An explicit value in the body still wins (operator override).

Validates `len(path) <= 255`, `len(checksum) <= 32`, `taken_datetime` present.

### 2.3 `POST /api/v1/ingest/dives` → `{workflow_id, run_id}`

Body `IngestDiveRequest` (§3.1). Starts `IngestDiveWorkflow` on
`fishsense_api_queue` with id `ingest-dive-{slugified dive_path}`,
`id_reuse_policy=ALLOW_DUPLICATE` (re-ingest/resume is the normal case; a
still-*running* ingest for the same folder raises `WorkflowAlreadyStartedError`
→ 409, which is correct).

### 2.4 `GET /api/v1/ingest/dives/{workflow_id}` → `IngestStatus`

`handle.describe()` for status/timestamps + `handle.query("progress")` for counts.
Query failure on a completed workflow degrades to describe-only rather than 500ing.

### 2.5 `GET /api/v1/ingest/dives` → `List[IngestStatus]`

Recent runs via `client.list_workflows('WorkflowType = "IngestDiveWorkflow"')`,
newest 25. Backs the portal's history table.

### 2.6 `POST /api/v1/ingest/dives/{workflow_id}/cancel` → 204

`handle.cancel()`. A cancelled ingest leaves the dive at LOW — inert. This is why cancel is safe
to expose.

---

### 2.7 Temporal client in fishsense-api — the boot-safety details

Three repo-specific traps make this less trivial than "add a client":

**(a) Dynaconf validates *everything* on first attribute access** — not lazily per
setting. The CLAUDE.md worker gotcha applies to the API too, so the new validators
must be **optional**, never `required=True`:

```python
Validator("temporal.host", cast=str, condition=validators.hostname),   # NOT required
Validator("temporal.port", cast=int, default=7233),
Validator("temporal.tls", cast=bool, default=False),
Validator("temporal.namespace", cast=str, default="default"),
Validator("temporal.client_cert", cast=str, condition=path_validator),
Validator("temporal.client_private_key", cast=str, condition=path_validator),
Validator("temporal.server_root_ca_cert", cast=str, condition=path_validator),
Validator("temporal.domain", cast=str),
```

A `required=True` here would fire on the *first* `settings.postgres.host` read in
`lifespan` and crash-loop the API on every deployment that hasn't set the temporal
block — including local dev and the whole test suite. Same shape as the api-worker's
optional `kubernetes.kubeconfig_path`, which no-ops when unset.

**(b) Import must not require settings.**
[test_import_without_settings.py](services/fishsense-api/tests/test_import_without_settings.py)
pins that importing the package + controllers works with no `settings.toml` in cwd.
So `temporal_client.py` reads settings **inside** the connect call, never at module
scope, and `ingest_controller.py` imports it without touching config. Add
`ingest_controller` to that test's pinned import list.

**(c) Connect lazily, once, under a lock.** `Client.connect` is async and the API is
async multi-request:

```python
_client: Client | None = None
_lock = asyncio.Lock()

async def get_temporal_client() -> Client:
    global _client
    if _client is None:
        async with _lock:
            if _client is None:                       # double-check under the lock
                if "host" not in settings.get("temporal", {}):
                    raise HTTPException(503, "Temporal is not configured")
                _client = await Client.connect(
                    f"{settings.temporal.host}:{settings.temporal.port}",
                    namespace=temporal_namespace(settings.temporal),
                    tls=build_tls_config(settings.temporal),
                )
    return _client
```

Reusing `fishsense_shared.build_tls_config` / `temporal_namespace` verbatim means no
second TLS implementation. `build_tls_config` takes the `settings.temporal` subtree
and returns `None` when `tls` is false, so local dev against
`temporal server start-dev` works unchanged.

**Unconfigured or unreachable → 503 on the four ingest routes only.** Every existing
route is untouched. That is the point of connecting lazily: a Temporal outage
degrades ingest rather than taking down the API.

**(d) Allowlist the workflow type.** The endpoint never accepts a workflow name from
the caller:

```python
_STARTABLE = {"IngestDiveWorkflow"}
```

"The API can start workflows" is the real widening here; constraining it to one
hardcoded type keeps the blast radius at "can start an ingest".

---

## 3. Contracts (`libs/fishsense-shared`)

Pydantic `BaseModel`s in a new `ingest_contracts.py`, re-exported from
`fishsense_shared.__init__`. Same rationale as `preprocess_contracts`: this is the
fishsense-api ↔ api-worker contract, and neither package may import the other.

### 3.1 `IngestDiveRequest`

**One request names one dive.** Not a crawl — the operator supplies the specific
folder, and that folder *is* the dive. This also matches the legacy semantic
(§4.1): a dive was always exactly one directory's worth of frames.

```python
dive_path: str                     # NAS path, relative to e4e_nas.raw_root_path
dive_name: str | None = None       # default: leaf directory name
camera_id: int | None = None       # override; else MakerNote serial -> Camera.serial_number
priority: Priority = Priority.HIGH
dive_slate_id: int | None = None
calibration_dive_id: int | None = None
self_calibrates: bool = False      # exactly one of this / calibration_dive_id
flip_dive_slate: bool = False
dry_run: bool = False
```

**No `verify_existing`** — see §4.6. It shipped as a declared field honoured by
no code and was removed rather than wired.

### 3.2 `IngestPreflight`, `IngestProgress`, `IngestReport`

`IngestPreflight` — `files[]` (path, size, `taken_datetime`, `artist`), `errors[]`,
`warnings[]`, `resolved_camera_id`, `total_bytes`. **Dry-run returns this and
writes nothing.**

There is no NAS browser in the portal (the API must never touch the NAS), so the
**dry run is the browser**: the operator types a path, previews exactly what would
be ingested, then confirms. The ranged-read trick (§0.2) is what keeps it cheap —
~1 MB/file instead of ~15 MB.

`IngestProgress` — the query-handler shape: `state`, `dive_id`, `total`, `scanned`,
`registered`, `skipped_existing`, `rejected`, `current_path`.

`IngestReport` — the workflow return value, superset of progress plus
`rejected_details[]` and `dive_datetime`.

---

## 4. Workflow + activity breakdown

`IngestDiveWorkflow(IngestDiveRequest) -> IngestReport`, api-worker,
`fishsense_api_queue`, **on-demand — no schedule** (nothing to add to
[test_schedule_registration.py](services/fishsense-api-workflow-worker/tests/test_schedule_registration.py)'s
stagger).

### 4.1 `list_dive_folder_activity(request) -> DiveFolderListing`

**Non-recursive**, by design and by precedent. The request names the dive folder;
the images are the `.ORF` files **directly inside it** (case-insensitive match).

The precedent matters here, because it is what tells us non-recursive is *correct*
rather than merely simpler. Spider (`discovery.py:238`) walked recursively but then
assigned `dive = image.parent.relative_to(data_root)` — so **a dive was always
exactly one directory**, and a nested folder became its own separate dive row, never
extra frames on the parent. Recursing and attaching children to the named dive would
therefore merge dives that are distinct rows in prod today.

So subdirectories are **not** ingested silently. Preflight lists any it finds
containing `.ORF`s and reports them:

> `101923_Alligator1_FSL06/` contains 47 `.ORF` files. Under the existing
> convention that is a **separate dive** — submit it as its own request.

That is the Olympus rollover case (the TG-6 wraps its counter at `PA199999` and
starts a child folder; see
[tools/scan_image_path_rollover.py](tools/scan_image_path_rollover.py)). Surfacing it
as "here is another dive to submit" keeps the operator in control and keeps one
request meaning one dive.

Side note from the same evidence: the 303 rollover rows that tool found carry *flat*
paths which 404. Spider's code could not have produced those, so they came from an
earlier ingest or a hand-edit — not from the behaviour we are matching. Not ours to
fix here.

### 4.2 `preflight_ingest_activity(request, listing) -> IngestPreflight`

Ranged 1 MB read per file → EXIF (§0.2 makes this ~1 MB instead of ~15 MB);
resolves the camera from the MakerNote serial (§0.5) and runs every check in §1.
Returns *all* errors at once, not first-wins. Non-retryable `ApplicationError` if
`errors` is non-empty. **Zero writes.** If `dry_run`, the workflow returns here.

Extra checks earned from the legacy code:

* **Multi-camera dive → hard failure** (spider only wrote a report file, §0.4).
  Mixed intrinsics silently corrupt stage 14, and one folder should be one rig.
* **Serial resolves to a known `Camera`, or fail** — no `Artist` fallback (§0.5).
  `Artist` disagreeing with the resolved camera's `name` is a warning.
* **Sub-directories containing `.ORF`s are reported, not ingested** (§4.1).
* **Duplicate detection — see §4.2.1, and NOT the MD5 aggregate.**

#### 4.2.1 Duplicate-dive detection: replacing the whole-dive MD5 aggregate

Spider detected duplicate dives with
`MD5(STRING_AGG(basename || ':' || image_md5, '' ORDER BY path))`
(`select_compute_dive_checksum_by_dive.sql`). You said it never worked as well as
you wanted, and the reason is structural — it is worth naming, because it rules out
"just port it and tune it":

* **All-or-nothing.** One extra, missing, or re-copied frame changes the digest
  completely. Two folders that share 54 of 55 frames look exactly as unrelated as
  two folders sharing none. Most real duplicates are *near*-duplicates.
* **Basename-sensitive.** It hashes `basename:md5`, so a rename — or the Olympus
  rollover renumbering — breaks the match even when the bytes are identical.
  Content-addressing that is defeated by filenames is self-defeating.
* **No similarity measure.** It answers yes/no, but the operational question is
  "how much of this already exists, and where", which is what decides whether the
  new rows land non-canonical.

Proposed replacement, two layers, cheap-first:

**Layer 1 — free, runs in the dry run: leaf-name collision.** Compare the dive
folder's leaf name against existing `Dive.path` leaves. This catches the exact prod
case (dives 64 and 66 are both `082929_FishModels_FSL07`) at the cost of one
`dives.get()` the preflight already makes. Warning only.

**Layer 2 — exact, runs after the scan, before the commit flag: content-set
containment.** The scan has already computed every checksum, so ask the API which of
them exist and where, then report per overlapping dive:

```
containment = |new ∩ existing_dive| / |new|
```

`1.0` = this folder is wholly contained in that dive; `0.87` = 48 of 55 frames are
already there. Robust to extra frames, missing frames, and renames, because it is a
set operation on content hashes with no filename and no ordering involved. It also
degrades gracefully: a partial overlap is *reported as* a partial overlap.

Presented to the operator before `priority` flips, e.g. *"48/55 frames already exist
in dive 64 (containment 0.87); those images will be created non-canonical."* A
warning, never a block — re-ingesting the same frames under a second dive path is
legitimate and already happened in prod. Optionally gate on an explicit
`allow_duplicates` acknowledgement when containment is above, say, 0.9.

Needs one new endpoint: `POST /api/v1/images/checksums/lookup` taking a checksum
list and returning `{checksum: [{image_id, dive_id, is_canonical}]}` — a batch form
of the existing `GET /api/v1/images/checksum/{checksum}`. Worth it regardless: it is
also the natural way to answer "have I seen these frames before" from any tool.

### 4.3 `create_dive_activity(request, preflight) -> int`

`dives.post` with `priority=LOW` **regardless of the request** and `dive_datetime`
= provisional max from preflight EXIF. Returns `dive_id`.

### 4.4 `scan_and_register_images_activity(dive_id, batch, camera_id) -> BatchResult`

Per file, in order: skip if `images.get(path=…)` exists (no download at all); else
`download_to` → **streaming `md5()` in 8192-byte chunks** (§0.1, matching
`get_file_checksum` and avoiding a 15 MB buffer per file) → EXIF `DateTime`
(0x0132) → `images.post`. `activity.heartbeat(index)` per file; a retry resumes
from the recorded index. Batch of 25 (`ingest.batch_size`).

Reuses the NAS-error classification from
[stage_raw_bytes_for_dive_activity](services/fishsense-api-workflow-worker/src/fishsense_api_workflow_worker/activities/stage_raw_bytes_for_dive_activity.py):
DSM 408 → non-retryable; 502/407/402 → propagate for Temporal's bounded jittered
retry. **No inner retry loop** — that is what tripped the NAS auto-block.

### 4.5 `finalize_dive_activity(dive_id, request, totals) -> IngestReport`

Refuses (non-retryable) if `rejected > 0` or `registered + skipped != total`.
Otherwise `dives.post` again with real `dive_datetime` = **max** taken_datetime
(§0.4), `name` = leaf directory, `camera_id`, `dive_slate_id`,
`calibration_dive_id`, and `priority` from the request — the commit flag. Also
carries the §4.2.1 containment report, so the duplicate picture is in front of the
operator at the moment of commit.

### 4.6 Verifying existing rows — a workflow, not an ingest flag

**Resolved differently from this plan's original design.** The plan called for an
`IngestDiveRequest.verify_existing` mode: for a folder whose rows already exist,
re-hash and report mismatches without writing.

The field shipped; the behaviour did not. It sat on the contract honoured by no
code, so an operator setting it got a normal ingest and no warning — strictly
worse than an absent flag, because it reads as a safety measure. **Removed**
(#618) rather than wired.

Re-hashing existing rows is `VerifyDiveChecksumsWorkflow` (one dive) and
`VerifyAllDivesChecksumsWorkflow` (every canonical dive), which are better tools
for the job than a mode inside ingest would have been:

* **read-only by construction**, with a test tripwire asserting the module's
  source contains no `upload` / `delete` / `post` / `put`. An ingest flag would
  have been one `if` away from writing.
* **findings, not failures** — a missing file or a mismatch is recorded and the
  run continues, because that is the question being asked. Ingest inverts this
  deliberately: it is trying to *do* something, so a missing file is a failure.
* **no second copy of the conventions.** Both call
  `nas_frames.file_checksum` / `read_taken_datetime`, so there is exactly one
  definition of each — which matters because both fail silently when wrong.
* it samples (`limit`), so "does the convention hold" costs ~20 GB rather than
  ~930 GB.

Run over all 272 canonical dives on 2026-08-17: **1,619 frames, zero checksum
disagreements.** Two dives disagreed on timestamps — see §0.1.

---

## 5. Files to add / change

### fishsense-api

| File | Why |
|---|---|
| `controllers/dive_controller.py` | add `POST /api/v1/dives/` (upsert-by-path + validation) |
| `controllers/image_controller.py` | add `POST /api/v1/dives/{dive_id}/images/` (upsert-by-path + server-side `is_canonical`) and `POST /api/v1/images/checksums/lookup` (batch, §4.2.1) |
| `controllers/ingest_controller.py` **(new)** | start / status / list / cancel |
| `controllers/__init__.py` | **route registry** — import the new controller or its routes silently never register |
| `temporal_client.py` **(new)** | lazy locked singleton; reuses `build_tls_config` + `temporal_namespace`; workflow-type allowlist; settings read inside the call, not at import (§2.7) |
| `config.py` | `[temporal]` validators — **all optional**, never `required=True` (§2.7a) |
| `tests/test_import_without_settings.py` | add `ingest_controller` to the pinned import list |
| `pyproject.toml` | `temporalio>=1.15.0` |

No new SQLModel, no migration, no `database.py` change — ingest writes existing
tables. No `test_sdk_drift.py` impact for the same reason.

### fishsense-api-sdk

| File | Why |
|---|---|
| `clients/dive_client.py` | `post(dive) -> int` |
| `clients/image_client.py` | `post(dive_id, image) -> int` |

Pydantic mirrors already exist and are field-complete, so drift stays green.

### fishsense-shared

| File | Why |
|---|---|
| `ingest_contracts.py` **(new)**, `__init__.py` | the api ↔ api-worker DTO contract |

### fishsense-api-workflow-worker

| File | Why |
|---|---|
| `nas.py` | `list_dir(path) -> list[NasEntry]`, `download_range(path, offset, length)` |
| `exif.py` **(new)** | stdlib reader: tag 0x0132 + Olympus MakerNote serial (§0.5/§0.7). Pillow cannot open ORF |
| `activities/list_dive_folder_activity.py` **(new)** | §4.1 — non-recursive listing + subfolder report |
| `activities/preflight_ingest_activity.py` **(new)** | §4.2 |
| `activities/create_dive_activity.py` **(new)** | §4.3 |
| `activities/scan_and_register_images_activity.py` **(new)** | §4.4 |
| `activities/finalize_dive_activity.py` **(new)** | §4.5 |
| `workflows/ingest_dive_workflow.py` **(new)** | orchestration + `progress` query |
| `worker.py` | register the workflow + 5 activities (no schedule) |
| `config.py` | `ingest.batch_size` (default 25) |

### fishsense-lite-web

| File | Why |
|---|---|
| `lib/ingest.ts` **(new)** | typed fetch wrappers, `safeId`-style validation on the workflow id before it enters a URL |
| `app/portal/ingest/page.tsx` **(new)** | **must call `auth()` itself** — there is no `middleware.ts`; `/portal` is gated per-page (a deliberate workaround for a Next 15.5 middleware-loader bug, pinned by `portal.integration.test.ts`) |
| `app/portal/ingest/ingest-form.tsx` **(new)** | client form: path, camera override, priority, calibration intent, dry-run/confirm, 3 s progress poll |
| `app/portal/ingest/actions.ts` **(new)** | server actions, each re-checking `auth()` (server actions are public endpoints — same pattern as [actions.ts](apps/fishsense-lite-web/app/portal/actions.ts)) |
| `app/portal/page.tsx` | link to the ingest page |

### deploy

| File | Why |
|---|---|
| `deploy/incus/compose.yml` | mount `/run/tenant/temporal:/run/tenant/temporal:ro` on `fishsense-api` |
| `deploy/incus/fishsense_api_volumes/config/settings.toml` | `[temporal]` → krg-prod, copied from the api-worker's stanza (same certs, same namespace) |
| `deploy/compose.local.yml` | local Temporal for the API so the integration tests can run |

---

## 6. Test list — failing first, in order

TDD per [CLAUDE.md](CLAUDE.md). Each bullet is a red test before its implementation.

**API — `test_dive_ingest_endpoints.py`, `test_image_ingest_endpoints.py`**
(in-memory sqlite, controller functions called directly, per
[test_calibration_source_endpoints.py](services/fishsense-api/tests/test_calibration_source_endpoints.py)):

1. `POST /dives/` creates and returns an id.
2. **`POST /dives/` twice with the same `path` returns the same id and leaves one
   row** ← the #347 regression; fails first against a blind merge.
3. `POST /dives/` 422s on: missing camera, camera without intrinsics,
   self-referential `calibration_dive_id`, `path` > 255.
4. `POST /dives/{id}/images/` creates; forces `dive_id` from the path.
5. **`POST` image twice with the same `path` → one row, same id.**
6. **First image with a checksum gets `is_canonical=True`; a second image with a
   different path and the same checksum gets `False`** ← the dives-64/66 rule.
6b. `POST /images/checksums/lookup` returns every dive holding each checksum, and
    `[]` for unknown ones (§4.2.1).
7. An explicit `is_canonical` in the body overrides the computed value.
8. Image 422s on `path` > 255 / missing `taken_datetime`.
9. `POST /ingest/dives` calls `start_workflow` with the right type, queue and id
   (mocked client); a `WorkflowAlreadyStartedError` surfaces as 409.
9b. **With no `[temporal]` config, every existing route still works and the four
    ingest routes return 503** — the boot-safety property (§2.7). Fails first
    against a `required=True` validator or a module-scope connect.
9c. `get_temporal_client` connects once under concurrent callers (asyncio lock).
9d. The allowlist rejects any workflow type but `IngestDiveWorkflow`.
10. `GET /ingest/dives/{id}` merges `describe()` + `query()`; a query failure on a
    finished run degrades instead of 500ing.

**SDK — `tests/clients/test_dive_client.py`, `test_image_client.py`**
(`asyncio_mode="auto"`, no decorator; clients used inside `async with`):

11. `dives.post` POSTs to `/api/v1/dives/` and returns the parsed int.
12. `images.post` POSTs to `/api/v1/dives/{id}/images/`.
13. Calling either outside `async with` raises `RuntimeError` (`ClientBase` contract).

**EXIF — `test_exif.py`:**

14. Parses `DateTime` (0x0132), `Artist`, `Model` from a committed ORF header fixture.
14b. **Extracts the Olympus MakerNote `SerialNumber`** — pins base-relative offsets
     *and* the type-13 Equipment pointer, the two bugs that made me wrongly call
     MakerNote parsing infeasible (§0.5).
15. **A file with no `DateTimeOriginal` returns `None` — never a default.**
16. Works on a truncated first-1 MB buffer (the ranged-read path).
17. Big-endian (`MM`) TIFF parses too.

**NAS — `test_nas_client.py` (extend):**

18. `list_dir` passes the path through and maps entries to `NasEntry`.
19. `download_range` forwards `offset`/`length` to `Client.download`.
20. **Tripwire: the ingest activity module imports no NAS *write* symbol** — mirrors
    the existing cleanup-activity tripwire. NAS stays read-only.

**Activities:**

21. `list_dive_folder` matches `.ORF` case-insensitively, is **non-recursive**, and
    **reports — does not ingest — a subdirectory containing `.ORF`s** (§4.1).
22. `preflight` returns **all** errors at once, not first-wins.
23. `preflight` fails when the camera has no intrinsics.
24. `preflight` fails when neither `calibration_dive_id` nor `self_calibrates` is given.
25. `preflight` fails on a >255-char path, naming the offender.
26. `preflight` resolves camera from the **MakerNote serial**; `camera_id` overrides.
26b. **`preflight` fails when the serial matches no `Camera` — it does NOT fall back
     to `Artist`** (§0.5). The anti-regression test for the camera decision.
26c. **`preflight` fails a dive whose frames span two serials** (§0.4).
26d. `preflight` warns when `Artist` disagrees with the resolved camera's `name`.
26e. `preflight` warns on a leaf-name collision with an existing dive (§4.2.1 L1).
26f. **Containment is computed as a set operation**: 48/55 shared checksums reports
     0.87 regardless of filenames or ordering — the property the MD5 aggregate
     lacked (§4.2.1 L2).
27. **`create_dive` forces `priority=LOW` even when the request says HIGH.**
28. `scan_and_register` skips an existing path **without downloading**.
29. `scan_and_register` heartbeats per file and resumes from heartbeat details.
30. `scan_and_register` classifies DSM 408 as non-retryable, 502 as retryable.
31. **A file with unreadable EXIF is rejected, not defaulted.**
31b. `taken_datetime` comes from **0x0132**, falls back to 0x9003 with a log line,
     and is UTC-stamped from the naive value (§0.3).
31c. **Checksum equals `md5` of the whole file** — asserted against the real ORF
     fixture, pinning §0.1's `get_file_checksum` equivalence.
32. **`finalize` refuses when `rejected > 0` — the dive stays LOW.**
33. `finalize` sets `dive_datetime` to the **max** taken_datetime (§0.4).

**Workflow — `test_ingest_dive_workflow.py`** (in-process Temporal env, as in
`test_preprocess_laser_images_parent_workflow.py`):

34. Happy path: activities called in order, report totals correct.
35. **`dry_run=True` returns the preflight and calls no write activity.**
36. Batching: 60 files → 3 activity calls at batch size 25.
37. **A crash in batch 2 leaves the dive at LOW and `finalize` uncalled.**
38. The `progress` query returns live counts mid-run.
39. Re-run over a fully-ingested folder: 0 downloads, dive still HIGH.

**Web — vitest:**

40. `lib/ingest.ts` sends Basic auth and parses the response (per `dives.test.ts`).
41. `lib/ingest.ts` rejects a malformed workflow id before building the URL.
42. Each server action throws when `auth()` returns no session.
43. Integration: signed-out `GET /portal/ingest` → 307 to sign-in (mirrors
    `portal.integration.test.ts` — the per-page gate is the only gate).

---

## 7. Part 2 — `Weasly Fish` is labelable but ungradeable

### 7.1 Confirmed

`_species_taxonomy_branch("Fish Model")` in the shipped XML
([create_species_label_studio_project_activity.py](services/fishsense-api-workflow-worker/src/fishsense_api_workflow_worker/activities/create_species_label_studio_project_activity.py)):

```
Weasly Fish, Snook, Grouper, Shark, Gray Anthias, Purple Angel, Yellow Anthias
```

`KNOWN_FISH_MODELS` in [views.py](services/fishsense-api/src/fishsense_api/views.py):

```
Snook, Grouper, Shark, Gray Anthias, Purple Angel, Yellow Anthias, Ruler
```

`Weasly Fish` is the only Fish Model choice with no reference row, exactly as you
said. `fish_model_measurement_accuracy` **inner**-joins `fishmodelreference` on
`Fish.name`, so its measurements would be written and then vanish from the view
with no error anywhere.

### 7.2 Changes

**a. `views.py` — add the entry.** `known_length_m: 0.30` per your figure, with a
comment marking it **provisional, uncalipered** and naming the diagnostic:

> Every other model reads *short* against truth by a consistent ladder ordered by
> fork depth (Purple Angel −2.2 %, Shark −2.6 %, Grouper −5.1 %, Snook −9.6 %). So
> a Weasly Fish landing inside roughly −2 % … −10 % is consistent with 30 cm being
> right. Readings materially **more negative** than that band say the 30 cm
> reference is too **long**; **positive** readings say it is too **short**.
> Recalipering settles it; the view is the instrument, not the authority.

**b. `notes` — the thickness experiment.** The column exists
([fish_model_reference.py](services/fishsense-api/src/fishsense_api/models/fish_model_reference.py))
but nothing has ever written it: both existing seed migrations insert only
`(name, known_length_m)`.

So this needs a `"notes"` key on the entry, formatted for machine re-parsing:

```
notes = "fork length provisional (uncalipered, operator estimate 2026-08-06); "
        "width_midbody_mm=<TBD>; width_caudal_peduncle_mm=<TBD>"
```

⚠️ **INPUT NEEDED — I do not have these two numbers and will not invent them.**
You asked for thickness to be an *independent calipered input*, not something
inferred from measurement error, which is exactly right and is exactly why I
cannot supply it. Give me mid-body width and caudal-peduncle width in mm and they
go straight in. Everything else in Part 2 ships without them; I would rather land
the row with `<TBD>` markers than a fabricated number, and the migration is
re-runnable for the fill-in.

*Compatibility check I ran:* adding a `notes` key to the shared
`KNOWN_FISH_MODELS` list means the two **already-shipped** migrations
(`e2c9a4f70b31`, `b4c81f60d7e2`) execute
`INSERT … VALUES (:name, :known_length_m)` with dicts that now carry an extra key.
I tested this — SQLAlchemy 2.0.51 ignores unbound keys in an executemany param
list, rows insert correctly. `test_alembic_upgrade.py` covers the regression.

**c. New alembic migration** (`down_revision` = current head; check `alembic
heads` at implementation time), following the seed-only-if-absent pattern from
both predecessors, but inserting `notes` as well:

```python
existing = {r[0] for r in bind.execute(sa.text("SELECT name FROM fishmodelreference"))}
to_insert = [m for m in KNOWN_FISH_MODELS if m["name"] not in existing]
if to_insert:
    bind.execute(sa.text(
        "INSERT INTO fishmodelreference (name, known_length_m, notes) "
        "VALUES (:name, :known_length_m, :notes)"),
        [{**m, "notes": m.get("notes")} for m in to_insert])
```

Seed-only-if-absent means it never clobbers a hand-corrected length — and it also
means **it will not backfill `notes` onto a row that already exists**. Since
`Weasly Fish` has no row in prod (0 labels, 0 Fish rows), this is fine today; a
later width correction needs its own `UPDATE` migration. Called out so nobody is
surprised. No view recreation needed — the accuracy view joins by name and picks
the row up automatically.

**d. Strengthen the guard so this cannot recur.**
`test_known_models_cover_the_labeled_taxonomy` currently asserts
`{Snook, Grouper, Shark, Purple Angel} <= names` — a hardcoded subset, which is
precisely why the gap survived. Replace with: parse the **actual shipped XML**,
extract every `Fish Model` child choice, assert each has a reference row.

The XML lives in the api-worker package and `fishsense-api` does not depend on it
(nor should it). Two options; I recommend the second:

* *Import it.* Works in CI (uv workspace, one shared `.venv`) but creates an
  undeclared cross-service dependency **and** trips the dynaconf gotcha — importing
  that activity chains into `config.settings`, which eagerly validates *every*
  `Validator` on first attribute access, so the api test suite would have to plumb
  `E4EFS_TEMPORAL__HOST`, `E4EFS_E4E_NAS__*`, etc.
* **Read it from disk and `ast`-parse the constant.** `REPO_ROOT =
  Path(__file__).resolve().parents[3]` — the same idiom
  [test_generated_sdk_models_freshness.py](services/fishsense-api/tests/test_generated_sdk_models_freshness.py)
  already uses — then `ast.parse` the module and `literal_eval` the
  `SPECIES_LABELING_CONFIG_XML` assignment, then `ElementTree` the XML. No import,
  no side effects, no config plumbing. A missing/renamed file fails loudly, which
  is the correct behaviour for a drift guard.

New tests: (i) every XML `Fish Model` choice has a reference row — **red before
the `Weasly Fish` entry lands**; (ii) `Weasly Fish` is present at 0.30 with a
provisional marker; (iii) its `notes` carries both width keys.

---

## 8. Rollout

Three PRs, in order, each independently deployable:

1. **`feat(api): dive + image create endpoints`** + SDK methods. No behaviour change
   until something calls them. Ships as `fishsense-api` minor → auto-deploy PR.
2. **`feat(api-worker): NAS folder ingest workflow`** + shared contracts + Temporal
   client in the API + compose/settings changes. The compose + settings.toml edits
   are committed IaC; the converge force-recreates, so they take effect
   ([CLAUDE.md](CLAUDE.md) "Committed config applies only because the converge
   force-recreates").
3. **`feat(web): portal ingest page`** — `apps/fishsense-lite-web` is `release-type:
   node`; the promote job bumps its compose pin like any other in-slot service.

Part 2 is a fourth, independent PR — no ordering constraint against the others.

**First production ingest**: dry-run first, always. Read the file list and the
resolved camera, confirm, then run for real. The dive sits at LOW until the last
image lands, so a bad first attempt costs NAS bandwidth and nothing else.

**SDK cascade:** PR 1 touches `libs/fishsense-api-sdk`, so release-please's
`auto-bump-sdk-consumers` job opens an `auto-bump/fishsense-api-sdk-<v>` PR. That
must merge before PR 2's worker image can see the new client methods.

---

## 9. Open items

1. ~~§0.1 checksum verification~~ — **closed** by
   `spider/backend.py:get_file_checksum`. Plain MD5 of the whole file. No prod
   access needed.
2. ~~§0.3 timezone convention~~ — **closed** by `spider/backend.py:get_image_date`.
   Naive EXIF 0x0132 → `TIMESTAMP` → migrated to `TIMESTAMPTZ` as UTC. Note the tag
   correction: 0x0132, not 0x9003.
3. ~~Camera key~~ — **decided: MakerNote serial**, matching existing rows. stdlib
   extraction verified on the fixture; no exiftool. Fails loudly rather than falling
   back (§0.5).
4. **No `POST /cameras` endpoint exists**, so a genuinely new rig blocks ingest until
   its `Camera` row is inserted DB-direct. Out of scope; flag if a new rig is due.
5. **Duplicate detection replaced** (§4.2.1) — leaf-name warning + content-set
   containment, instead of the whole-dive MD5 aggregate. **Threshold still open,
   and the prod numbers argue against 0.9** (see §0.9): a folder that is 60 %
   already-present is the common case, not the exception, and that is exactly
   when an operator wants telling. Suggest 0.5, or reporting containment always
   and gating the acknowledgement on 1.0 (wholly contained).
6. **Weasly Fish widths** — mid-body and caudal-peduncle, in mm.
7. **Weasly Fish 30 cm** — provisional; §7.2's diagnostic band tells you when to
   recaliper.
8. ~~Temporal-client-in-the-API vs. polled table~~ — **decided: client in the API.**
   Boot-safety constraints in §2.7; the load-bearing one is that the new `[temporal]`
   validators must be optional, or the API crash-loops anywhere the block is unset.
9. **Follow-up, deliberately out of scope:** the ingest already holds each file's
   bytes in memory, so it could stage them to Garage `raw/{checksum}.ORF` and save
   stage 0.1 a second 7.5 GB download. Real win, but it couples ingest to the
   object store and to JPEG-retention policy. Separate decision.
10. **Follow-up:** existing `Dive.camera_id` values are unreliable (§0.4 — the
   `images[-1]` bug, now confirmed against the migration source). Worth an audit;
   not this change.
11. ~~Archive the legacy repos~~ — **done 2026-08-06.**
    `fishsense-data-processing-worker` was already archived;
    `fishsense-data-processing-spider` archived at your request. Both stay readable;
    spider's 7 open issues are now locked to further comment. Reversible.
    Two consequences worth naming:
    * **Archiving is not decommissioning.** If a spider instance is still deployed
      anywhere, it keeps crawling and writing to the *legacy* `fabricant-prod`
      Postgres. That DB stopped being the source of truth at `9e5bc64`, so it would
      not corrupt fishsense-api — but it would burn NAS bandwidth against the same
      shares this ingest reads, and the fragile FileStation backend is the
      contended resource (krg-infra#501). **Worth confirming it is actually
      switched off before the first production ingest run.**
    * These conventions now live only in §0 here, the memory entry, and read-only
      GitHub. They belong in CLAUDE.md when this lands — the "Notebook port status"
      table has no stage-0 row, and this is it.
12. **A `MIGRATED_TO_MONOREPO.md`-style notice** was not added to either repo before
    archiving — `worker` had already been archived bare, so I matched that. If you
    want pointers in their READMEs it means unarchive → push → re-archive; say the
    word and I will.
