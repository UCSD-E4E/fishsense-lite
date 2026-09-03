"""Read-only offline validation of model-predicted head/tail keypoints.

Background
----------
`fishsense_core.fish` ships two models that, together, could seed the
stage-5.1 Label Studio head/tail tasks with pre-annotations the way
`LaserPrediction` already seeds the laser ones:

    FishSegmentation.inference(bgr)   -> (H,W) uint8 instance label map
    FishHeadTailDetector.find_head_tail_img(mask) -> (head_xy, tail_xy)

The proposed gate is "only use a fish the laser is on": look up the
instance id under the image's validated `LaserLabel` and run the
head/tail detector on that instance alone. Because `inference` returns
instance ids rather than a binary mask, that gate is a single array
lookup — and it is the whole reason this is worth testing, since the
segmenter's failure mode on cluttered pool dives is confident false
positives on divers' legs and swim shorts.

This script answers whether the pair is good enough to put in front of
labelers, **before** anything is written to Label Studio. That ordering
is the lesson of the slate detector: it shipped 2026-08-02 behind an
ECC >= 0.80 acceptance gate, and was shut down the next day because the
gate did not transfer out of distribution — pool dives produced
high-ECC *false* fits. An offline pass against the existing human label
corpus is the cheap version of that discovery.

What it measures
----------------
Every image with BOTH a completed non-superseded `HeadTailLabel` and a
valid `LaserLabel` is a free oracle: the human keypoints are ground
truth, in the same rectified-JPEG pixel space the prediction lands in
(`sync_headtail_labels_*` and `sync_laser_labels_*` both convert Label
Studio percentages via `original_width/height`, and `cv2.undistort`
preserves image size, so laser pixels, head/tail pixels and the stage
5.1 JPEG are all one coordinate system).

Per image it records:

  * **coverage** — did the gate produce a prediction at all, and if
    not, which of the three abstentions fired: nothing detected, the
    laser landed on no instance, or the detector failed on the mask.
    Abstaining is a *good* outcome relative to seeding a wrong point;
    the report keeps them separate rather than folding them into an
    error rate.
  * **orientation** — `find_head_tail_img` claims to return head
    first (it is not raw PCA; there is a peduncle stage). Both
    assignments are scored and the better one kept, so a systematic
    head/tail swap shows up as a low `orientation_ok` rate instead of
    silently doubling the position error.
  * **position error** — head and tail distance to the human
    keypoints, in pixels and as a percentage of the human fish length.
    Normalizing matters because a 40 px miss on a frame-filling snook
    and on a distant fish are not the same mistake.
  * **length error** — signed `(pred_len - human_len) / human_len`.
    This is the one that reaches `Measurement`, and it is where the
    fork-versus-tail-tip concern shows up: the LS label is `Fork`, and
    every reference length in `fishmodelreference` is a fork length, so
    a detector returning the caudal-lobe tip reads systematically long.
  * **what the gate buys** — whether the laser instance is also the
    largest instance, i.e. whether `inference_single` would have picked
    the same fish. The difference is the gate's value in one number.

Limits worth stating before anyone quotes the output
----------------------------------------------------
Human head/tail labels are not error-free themselves, so the reported
error is prediction error *plus* label noise, and the floor is not
zero. And this compares against what labelers drew, not against
physical truth — a systematic labeling convention error would be
invisible here and is exactly what the fish-model reference lengths
(not this script) are for.

Credentials
-----------
Read from the environment, never from a config file, so no secret is
read into a transcript:

    export FISHSENSE_DSN='postgresql://user:pass@host:5432/fishsense'
    export FISHSENSE_S3_ACCESS_KEY=...      # read on the labels bucket
    export FISHSENSE_S3_SECRET_KEY=...

`FISHSENSE_DSN` is happiest pointed at a **restored backup** rather than
prod: the manifest step is the only thing that touches a database, it is
pure SELECT, and a local restore removes the network path entirely.

The S3 pair is only needed to *fill* the JPEG cache. `--cache-dir` is
addressed by checksum, so a directory of `{checksum}.JPG` files fetched
by any other means works, and `predict` never contacts the object store.

The script issues SELECTs and S3 GETs only. It writes nothing to the
database, to Garage, or to Label Studio.

Run
---
    # 1. what is there to test?
    uv run python tools/validate_headtail_predictions.py manifest \
        --out ht_manifest.csv --per-dive 40

    # 2. download + infer (the slow step; ~2s/frame on CPU + download)
    uv run python tools/validate_headtail_predictions.py predict \
        --manifest ht_manifest.csv --out ht_results.csv

    # 3. read the answer (re-runnable without re-inferring)
    uv run python tools/validate_headtail_predictions.py report \
        --results ht_results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Physical Garage layout for the stage-5.1 JPEG — the exact bytes the
# labeler saw. Mirrors `fishsense_shared.object_store.jpeg_key`, spelled
# out here so the tool runs without importing a worker package (and so
# without tripping Dynaconf's eager validation).
HEADTAIL_JPEG_FOLDER = "preprocess_headtail_jpeg"
DEFAULT_ENDPOINT = "https://s3.e4e.ucsd.edu"
DEFAULT_REGION = "garage"
DEFAULT_LABELS_BUCKET = "labels-fishsense-lite"
DEFAULT_LABELS_PREFIX = "fishsense-lite"

# botocore surfaces a missing key differently for GetObject vs HeadObject.
# Mirrors `fishsense_shared.object_store.NOT_FOUND_CODES`.
NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})

MANIFEST_FIELDS = [
    "image_id",
    "dive_id",
    "dive_name",
    "checksum",
    "raw_path",
    "head_x",
    "head_y",
    "tail_x",
    "tail_y",
    "laser_points",
    "n_lasers",
    "camera_matrix",
    "distortion_coefficients",
]

RESULT_FIELDS = MANIFEST_FIELDS + [
    "frame_source",
    "status",
    "n_instances",
    "laser_instance",
    "laser_instance_is_largest",
    "instance_area_px",
    "pred_head_x",
    "pred_head_y",
    "pred_tail_x",
    "pred_tail_y",
    "orientation_ok",
    "err_head_px",
    "err_tail_px",
    "human_len_px",
    "pred_len_px",
    "err_head_pct",
    "err_tail_pct",
    "len_err_pct",
    "human_head_in_mask",
    "human_tail_in_mask",
]


# --------------------------------------------------------------------------
# manifest: what the database offers as ground truth
# --------------------------------------------------------------------------

# One row per image. Head/tail is deduped to the most recently updated
# qualifying label because an image can carry a row per LS project (the
# legacy shared projects plus its per-dive one). Lasers are aggregated
# rather than joined, because 461 prod images carry two valid laser
# labels and joining would multiply the row out — the predictor tries
# each point and takes the first that lands on a fish.
_MANIFEST_SQL = """
WITH ht AS (
    SELECT DISTINCT ON (h.image_id)
           h.image_id, h.head_x, h.head_y, h.tail_x, h.tail_y
    FROM headtaillabel h
    WHERE h.completed
      AND NOT COALESCE(h.superseded, false)
      AND h.head_x IS NOT NULL AND h.head_y IS NOT NULL
      AND h.tail_x IS NOT NULL AND h.tail_y IS NOT NULL
    ORDER BY h.image_id, h.updated_at DESC NULLS LAST, h.id DESC
),
las AS (
    SELECT l.image_id,
           string_agg(l.x || ',' || l.y, ';' ORDER BY l.id) AS laser_points,
           count(*) AS n_lasers
    FROM laserlabel l
    WHERE l.completed
      AND NOT COALESCE(l.superseded, false)
      AND l.x IS NOT NULL AND l.y IS NOT NULL
    GROUP BY l.image_id
)
SELECT i.id AS image_id,
       d.id AS dive_id,
       COALESCE(d.name, '') AS dive_name,
       i.checksum,
       i.path AS raw_path,
       ht.head_x, ht.head_y, ht.tail_x, ht.tail_y,
       las.laser_points, las.n_lasers,
       ci.camera_matrix::text AS camera_matrix,
       ci.distortion_coefficients::text AS distortion_coefficients
FROM image i
JOIN dive d  ON d.id = i.dive_id
JOIN ht      ON ht.image_id = i.id
JOIN las     ON las.image_id = i.id
LEFT JOIN cameraintrinsics ci ON ci.camera_id = d.camera_id
WHERE i.is_canonical
ORDER BY d.id, i.id
"""


def build_manifest(args: argparse.Namespace) -> int:
    """SELECT the oracle set and write it to CSV. Read-only."""
    import psycopg

    dsn = os.environ.get("FISHSENSE_DSN")
    if not dsn:
        sys.exit(
            "Set FISHSENSE_DSN, e.g.\n"
            "  export FISHSENSE_DSN='postgresql://user:pass@host:5432/fishsense'"
        )

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(_MANIFEST_SQL)
            cols = [c.name for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    total = len(rows)
    if args.dive_id:
        wanted = set(args.dive_id)
        rows = [r for r in rows if r["dive_id"] in wanted]

    # Cap per dive before the global limit so a single heavily-labelled
    # dive cannot stand in for the corpus — the failure modes seen so
    # far are dive-shaped (pool clutter, model props), not frame-shaped.
    #
    # Sampled evenly across each dive rather than taking the first N: a
    # dive's opening frames are not representative of it (setup shots,
    # slate frames, the diver still settling), so head-of-list sampling
    # would quietly measure the wrong population.
    if args.per_dive:
        grouped: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            grouped[r["dive_id"]].append(r)
        sampled = []
        for dive_id in sorted(grouped):
            drows = grouped[dive_id]
            if len(drows) <= args.per_dive:
                sampled.extend(drows)
                continue
            step = len(drows) / args.per_dive
            sampled.extend(
                drows[min(len(drows) - 1, int(i * step))]
                for i in range(args.per_dive)
            )
        rows = sampled

    if args.limit:
        rows = rows[: args.limit]

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in MANIFEST_FIELDS})

    if args.checksums_out:
        Path(args.checksums_out).write_text(
            "".join(f"{r['checksum']}\n" for r in rows), encoding="utf-8"
        )
        print(f"wrote {len(rows)} checksums -> {args.checksums_out}")

    dives = sorted({r["dive_id"] for r in rows})
    print(f"{total} images in the corpus with both a valid laser and head/tail label")
    print(f"wrote {len(rows)} rows across {len(dives)} dives -> {out}")
    if dives:
        print(f"dives: {', '.join(str(d) for d in dives)}")
    return 0


# --------------------------------------------------------------------------
# predict: download the stage-5.1 JPEG, segment, gate on the laser
# --------------------------------------------------------------------------


def _s3_client(args: argparse.Namespace):
    """Garage needs path-style addressing and SigV4 — same construction as
    `fishsense_shared.object_store.build_s3_client`."""
    import boto3
    from botocore.config import Config

    access = os.environ.get("FISHSENSE_S3_ACCESS_KEY")
    secret = os.environ.get("FISHSENSE_S3_SECRET_KEY")
    if not (access and secret):
        # Not fatal: a fully-populated `--cache-dir` needs no object store
        # at all, which is the whole point of the cache being addressed by
        # checksum. `_fetch_jpeg` fails only if it actually has to fetch.
        return None
    return boto3.client(
        "s3",
        endpoint_url=args.endpoint_url,
        region_name=args.region,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def _fetch_jpeg(s3, args: argparse.Namespace, checksum: str, cache: Path) -> bytes | None:
    """Return the stage-5.1 JPEG bytes, using the local cache when present.

    Returns None when the object is absent, which is an ordinary outcome:
    stage 5.1 only writes JPEGs for images it was dispatched for, and the
    head/tail corpus reaches back to the legacy shared projects.
    """
    local = cache / f"{checksum}.JPG"
    if local.exists():
        return local.read_bytes()

    if s3 is None:
        return None

    from botocore.exceptions import ClientError

    prefix = args.labels_prefix.strip("/")
    base = f"{HEADTAIL_JPEG_FOLDER}/{checksum}.JPG"
    key = f"{prefix}/{base}" if prefix else base
    try:
        body = s3.get_object(Bucket=args.labels_bucket, Key=key)["Body"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in NOT_FOUND_CODES:
            return None
        raise
    try:
        data = body.read()
    finally:
        body.close()

    if not args.no_cache:
        local.write_bytes(data)
    return data


def _fetch_frame(s3, args, row: dict, cache: Path) -> tuple[bytes | None, str]:
    """Resolve one image's rectified frame, and say where it came from.

    Order is deliberate: the cache first, then the Garage JPEG (the exact
    bytes a labeler saw), then a local rectify from the NAS raw. The
    source is recorded per row because the two are equivalent only if the
    stored intrinsics still match the ones the JPEG was made with —
    `check_rectify_parity` measures that, and `frame_source` is what lets
    the report be re-read if it ever stops holding.
    """
    local = cache / f"{row['checksum']}.JPG"
    if local.exists():
        return local.read_bytes(), "cache"

    data = _fetch_jpeg(s3, args, row["checksum"], cache)
    if data is not None:
        return data, "garage"

    if args.raw_root:
        data = _rectify_from_raw(row, args.raw_root)
        if data is not None:
            if not args.no_cache:
                local.write_bytes(data)
            return data, "raw_rectified"

    return None, "missing"


def _rectify_from_raw(row: dict, raw_root: str) -> bytes | None:
    """Reproduce the stage-5.1 frame locally from the NAS `.ORF`.

    Garage only holds a JPEG for images stage 5.1 has actually processed
    — 350 of the 16,987-image oracle set, because the head/tail label
    corpus long predates both stage 5.1 and the Garage migration. The
    raw files are still on the NAS, and
    `RectifiedImage(RawImage(bytes), intrinsics).data` is byte-for-byte
    what stage 5.1 encodes (pinned by its notebook-parity test), so
    rectifying locally reaches the rest of the corpus without changing
    what is being measured.

    Returns None when the raw file or the dive's intrinsics are missing.
    """
    import json

    import cv2
    import numpy as np
    from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
    from fishsense_core.image.raw_image import RawImage
    from fishsense_core.image.rectified_image import RectifiedImage

    if not (row.get("raw_path") and row.get("camera_matrix")):
        return None
    path = Path(raw_root) / row["raw_path"]
    if not path.exists():
        return None

    intrinsics = CameraIntrinsics(
        camera_matrix=np.array(json.loads(row["camera_matrix"]), dtype=float),
        distortion_coefficients=np.array(
            json.loads(row["distortion_coefficients"]), dtype=float
        ),
        camera_id=None,
    )
    rectified = RectifiedImage(RawImage(path.read_bytes()), intrinsics)
    ok, encoded = cv2.imencode(".jpg", rectified.data)
    return encoded.tobytes() if ok else None


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _parse_laser_points(raw: str) -> list[tuple[float, float]]:
    points = []
    for chunk in (raw or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, _, y = chunk.partition(",")
        points.append((float(x), float(y)))
    return points


def _score_row(row: dict, labels, instance: int, head, tail) -> dict:
    """Fill in the comparison columns for one successful prediction.

    Both head/tail assignments are scored and the better kept: a
    systematic swap should surface as an orientation statistic, not as
    inflated position error.
    """
    import numpy as np

    hx, hy = float(row["head_x"]), float(row["head_y"])
    tx, ty = float(row["tail_x"]), float(row["tail_y"])
    phx, phy = float(head[0]), float(head[1])
    ptx, pty = float(tail[0]), float(tail[1])

    direct = _dist(phx, phy, hx, hy) + _dist(ptx, pty, tx, ty)
    swapped = _dist(phx, phy, tx, ty) + _dist(ptx, pty, hx, hy)
    orientation_ok = direct <= swapped
    if not orientation_ok:
        phx, phy, ptx, pty = ptx, pty, phx, phy

    err_head = _dist(phx, phy, hx, hy)
    err_tail = _dist(ptx, pty, tx, ty)
    human_len = _dist(hx, hy, tx, ty)
    pred_len = _dist(phx, phy, ptx, pty)

    height, width = labels.shape
    mask = labels == instance

    def _inside(x: float, y: float) -> bool:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < width and 0 <= yi < height):
            return False
        return bool(mask[yi, xi])

    return {
        "status": "predicted",
        "laser_instance": instance,
        "instance_area_px": int(np.count_nonzero(mask)),
        # Reported in the detector's own order, so the raw output stays
        # auditable after the assignment swap above.
        "pred_head_x": round(float(head[0]), 1),
        "pred_head_y": round(float(head[1]), 1),
        "pred_tail_x": round(float(tail[0]), 1),
        "pred_tail_y": round(float(tail[1]), 1),
        "orientation_ok": int(orientation_ok),
        "err_head_px": round(err_head, 1),
        "err_tail_px": round(err_tail, 1),
        "human_len_px": round(human_len, 1),
        "pred_len_px": round(pred_len, 1),
        "err_head_pct": round(100 * err_head / human_len, 2) if human_len else "",
        "err_tail_pct": round(100 * err_tail / human_len, 2) if human_len else "",
        "len_err_pct": (
            round(100 * (pred_len - human_len) / human_len, 2) if human_len else ""
        ),
        "human_head_in_mask": int(_inside(hx, hy)),
        "human_tail_in_mask": int(_inside(tx, ty)),
    }


def _laser_instance(labels, laser_points: str) -> int:
    """The gate: instance id under the first laser point that lands on a
    fish, or 0 when none does.

    An image may carry more than one valid laser label (461 prod images
    do), and first-hit-wins mirrors what a real stage would do rather
    than trying to be clever about which dot is "the" dot.
    """
    height, width = labels.shape
    for lx, ly in _parse_laser_points(laser_points):
        xi, yi = int(round(lx)), int(round(ly))
        if 0 <= xi < width and 0 <= yi < height and labels[yi, xi]:
            return int(labels[yi, xi])
    return 0


def _predict_one(row: dict, data: bytes | None, segmentation, detector) -> dict:
    """Score one manifest row. Returns only the result columns.

    Every early return is an abstention with its own status, because
    abstaining is a *good* outcome relative to seeding a wrong keypoint
    and the report needs to tell them apart.
    """
    import cv2
    import numpy as np

    if data is None:
        return {"status": "no_jpeg"}

    # cv2 decodes to BGR, which is what the segmenter wants — feeding it
    # RGB measurably degrades detection. The stage-5.1 JPEG was itself
    # encoded from `RectifiedImage.data` (BGR), so this round-trips the
    # labeler's exact pixels.
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"status": "bad_jpeg"}

    labels = segmentation.inference(bgr)
    instances = [int(v) for v in np.unique(labels) if v]
    if not instances:
        return {"status": "no_detections", "n_instances": 0}

    areas = {v: int(np.count_nonzero(labels == v)) for v in instances}
    largest = max(areas, key=areas.__getitem__)
    base = {"n_instances": len(instances)}

    hit = _laser_instance(labels, row["laser_points"])
    if not hit:
        return base | {"status": "laser_off_all_fish", "laser_instance_is_largest": 0}

    base["laser_instance_is_largest"] = int(hit == largest)
    mask = ((labels == hit).astype(np.uint8)) * 255
    # The detector is a PyO3 native call whose error surface is not a
    # documented exception hierarchy, and one unfittable mask must not
    # abort a several-hundred-image sweep.
    try:
        head, tail = detector.find_head_tail_img(mask)
    # pylint: disable-next=broad-exception-caught
    except Exception as exc:
        print(f"  image {row['image_id']}: find_head_tail_img: {exc}")
        return base | {
            "status": "headtail_failed",
            "laser_instance": hit,
            "instance_area_px": areas[hit],
        }

    return base | _score_row(row, labels, hit, head, tail)


def predict(args: argparse.Namespace) -> int:
    """Run the gated predictor over the manifest and write per-image results."""
    from fishsense_core.fish import FishHeadTailDetector, FishSegmentation

    manifest = Path(args.manifest)
    with manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit(f"{manifest} has no rows")

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    s3 = _s3_client(args)

    segmentation = FishSegmentation()
    segmentation.load_model()
    detector = FishHeadTailDetector()
    print(f"segmentation provider: {segmentation.active_provider()}", flush=True)

    out = Path(args.out)

    # Resume support. A full sweep is hours of rawpy decode, and the run has
    # no checkpoint of its own — an interrupted one (a suspended laptop, an
    # OOM) otherwise starts from zero. Rows already scored are skipped and
    # the file is appended to, so re-issuing the same command finishes the
    # job rather than repeating it.
    done: set[str] = set()
    if args.resume and out.exists():
        with out.open(newline="", encoding="utf-8") as fh:
            done = {r["image_id"] for r in csv.DictReader(fh) if r.get("status")}
        rows = [r for r in rows if r["image_id"] not in done]
        print(f"resuming: {len(done)} already scored, {len(rows)} to go", flush=True)
        if not rows:
            return 0

    mode = "a" if (args.resume and out.exists() and done) else "w"
    blanks = {k: "" for k in RESULT_FIELDS if k not in MANIFEST_FIELDS}
    with out.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        for i, row in enumerate(rows, 1):
            data, source = _fetch_frame(s3, args, row, cache)
            writer.writerow(
                dict(row)
                | blanks
                | {"frame_source": source}
                | _predict_one(row, data, segmentation, detector)
            )
            if i % args.progress_every == 0:
                fh.flush()
                print(f"  {i}/{len(rows)}", flush=True)

    print(f"wrote {len(rows)} rows -> {out}")
    return 0


# --------------------------------------------------------------------------
# visualize: look at the frames, because statistics hide mask fragmentation
# --------------------------------------------------------------------------


def _mark(img, x, y, color, label: str) -> None:
    """Draw one labelled keypoint on a frame, in place."""
    import cv2

    point = (int(float(x)), int(float(y)))
    cv2.circle(img, point, 30, color, -1)
    cv2.putText(
        img,
        label,
        (point[0] + 40, point[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.2,
        color,
        5,
    )


def visualize(args: argparse.Namespace) -> int:
    """Render side-by-side keypoint overlays for a sample of rows.

    Aggregate error cannot distinguish a tight mask whose fork point sits
    slightly forward from a mask fragmented across the fish — both can land
    at the same pixel error while meaning opposite things about whether the
    stage is worth building. Green is the human label, magenta the
    prediction.
    """
    import cv2
    import numpy as np

    with Path(args.results).open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["status"] == args.status]
    if not rows:
        sys.exit(f"no rows with status={args.status}")

    if args.worst and args.status == "predicted":
        rows.sort(key=lambda r: -float(r["err_head_pct"] or 0))
    rows = rows[: args.limit]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    s3 = _s3_client(args)

    for row in rows:
        data, _ = _fetch_frame(s3, args, row, cache)
        if data is None:
            continue
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

        _mark(img, row["head_x"], row["head_y"], (0, 220, 0), "snout(human)")
        _mark(img, row["tail_x"], row["tail_y"], (0, 220, 0), "fork(human)")
        if row["pred_head_x"]:
            _mark(img, row["pred_head_x"], row["pred_head_y"], (255, 0, 255), "snout(pred)")
            _mark(img, row["pred_tail_x"], row["pred_tail_y"], (255, 0, 255), "fork(pred)")
        for lx, ly in _parse_laser_points(row["laser_points"]):
            cv2.drawMarker(
                img, (int(lx), int(ly)), (0, 255, 255), cv2.MARKER_CROSS, 90, 8
            )

        name = (
            f"dive{row['dive_id']}_img{row['image_id']}"
            f"_head{row['err_head_pct'] or 'na'}pct.jpg"
        )
        cv2.imwrite(str(out / name), cv2.resize(img, (1400, 1052)))
        print(f"  {name}")

    print(f"wrote {len(rows)} overlays -> {out}")
    return 0


# --------------------------------------------------------------------------
# report: read the answer
# --------------------------------------------------------------------------


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:5.1f}%" if whole else "    -"


def _quantiles(values: list[float], qs=(50, 90, 95)) -> dict[int, float]:
    if not values:
        return {q: float("nan") for q in qs}
    ordered = sorted(values)
    out = {}
    for q in qs:
        idx = min(len(ordered) - 1, int(round((q / 100) * (len(ordered) - 1))))
        out[q] = ordered[idx]
    return out


def _col(rows: list[dict], name: str) -> list[float]:
    """Numeric column over scored rows, skipping blanks."""
    return [float(r[name]) for r in rows if r[name] not in ("", None)]


def _report_quality(predicted: list[dict]) -> None:
    """The three yes/no properties: ordering, what the gate bought, and
    whether the human keypoints even land on the predicted fish."""
    orientation = [int(r["orientation_ok"]) for r in predicted]
    gate = [
        int(r["laser_instance_is_largest"])
        for r in predicted
        if r["laser_instance_is_largest"] not in ("", None)
    ]
    head_in = [int(r["human_head_in_mask"]) for r in predicted]
    tail_in = [int(r["human_tail_in_mask"]) for r in predicted]

    print(f"\n=== on the {len(predicted)} predicted images ===")
    print(
        f"  head-first ordering correct   {sum(orientation):6d}  "
        f"{_pct(sum(orientation), len(orientation))}"
    )
    print(
        f"  laser fish == largest fish    {sum(gate):6d}  "
        f"{_pct(sum(gate), len(gate))}   "
        f"(the gate changed the answer on {len(gate) - sum(gate)})"
    )
    print(
        f"  human snout inside mask       {sum(head_in):6d}  "
        f"{_pct(sum(head_in), len(head_in))}"
    )
    print(
        f"  human fork inside mask        {sum(tail_in):6d}  "
        f"{_pct(sum(tail_in), len(tail_in))}"
    )


def _report_errors(predicted: list[dict]) -> None:
    """Position and length error distributions."""
    print("\n=== error distributions (percent of human fish length) ===")
    print(f"  {'metric':22s} {'p50':>9s} {'p90':>9s} {'p95':>9s}")
    for name in ("err_head_pct", "err_tail_pct"):
        q = _quantiles(_col(predicted, name))
        print(f"  {name:22s} {q[50]:9.1f} {q[90]:9.1f} {q[95]:9.1f}")

    # Length error is signed, and the sign is the whole fork-vs-tail-tip
    # question, so it is reported both ways: absolute for magnitude, signed
    # for bias.
    lengths = _col(predicted, "len_err_pct")
    q = _quantiles([abs(v) for v in lengths])
    print(f"  {'abs(len_err_pct)':22s} {q[50]:9.1f} {q[90]:9.1f} {q[95]:9.1f}")
    signed = _quantiles(lengths, qs=(10, 50, 90))
    print(
        f"\n  signed len_err_pct: p10 {signed[10]:+.1f}  median {signed[50]:+.1f}  "
        f"p90 {signed[90]:+.1f}"
    )
    print(
        "  (a consistently positive median is the caudal-tip-not-fork "
        "signature; the\n   LS label is `Fork` and every reference length "
        "in fishmodelreference is a fork\n   length, so a positive bias "
        "would reach Measurement if labelers accept it)"
    )

    print("\n=== within-threshold rates (percent of human fish length) ===")
    for thresh in (2.0, 5.0, 10.0):
        head_ok = sum(1 for v in _col(predicted, "err_head_pct") if v <= thresh)
        tail_ok = sum(1 for v in _col(predicted, "err_tail_pct") if v <= thresh)
        print(
            f"  <= {thresh:4.1f}%   snout {head_ok:5d} {_pct(head_ok, len(predicted))}"
            f"   fork {tail_ok:5d} {_pct(tail_ok, len(predicted))}"
        )


def _report_per_dive(rows: list[dict]) -> None:
    """Break the same numbers out per dive.

    Worth reading before any aggregate: the failure modes seen so far are
    dive-shaped (pool clutter, rigid model props), so a healthy corpus
    median can sit on top of a dive that fails outright.
    """
    print("\n=== per dive ===")
    per_dive: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_dive[f"{r['dive_id']} {r['dive_name']}"].append(r)
    print(
        f"  {'dive':38s} {'n':>5s} {'pred':>6s} {'orient':>7s} "
        f"{'p50 head':>9s} {'p50 len':>9s}"
    )
    for dive, drows in sorted(per_dive.items(), key=lambda kv: int(kv[0].split()[0])):
        dpred = [r for r in drows if r["status"] == "predicted"]
        if dpred:
            orient = sum(int(r["orientation_ok"]) for r in dpred) / len(dpred)
            head_q = _quantiles(_col(dpred, "err_head_pct"))[50]
            len_q = _quantiles(_col(dpred, "len_err_pct"))[50]
            stats = f"{100 * orient:6.0f}% {head_q:9.1f} {len_q:+9.1f}"
        else:
            stats = f"{'-':>7s} {'-':>9s} {'-':>9s}"
        print(f"  {dive[:38]:38s} {len(drows):5d} {len(dpred):6d} {stats}")


def report(args: argparse.Namespace) -> int:
    """Summarize a results CSV. Pure reading — safe to re-run."""
    with Path(args.results).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("no rows")

    total = len(rows)
    predicted = [r for r in rows if r["status"] == "predicted"]

    print(f"\n=== coverage ({total} images with human head/tail + valid laser) ===")
    for status, n in Counter(r["status"] for r in rows).most_common():
        print(f"  {status:20s} {n:6d}  {_pct(n, total)}")

    # Where the pixels came from. `garage` and `raw_rectified` are the same
    # bytes (verified byte-for-byte on one image per dive), so this is a
    # provenance record rather than a caveat — but it is the first thing to
    # re-check if the two ever diverge.
    sources = Counter(r.get("frame_source", "") for r in rows)
    if any(sources):
        print("  frame source:", ", ".join(f"{k}={v}" for k, v in sources.most_common()))

    if not predicted:
        print("\nno predictions to score")
        return 0

    _report_quality(predicted)
    _report_errors(predicted)
    _report_per_dive(rows)
    return 0


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("manifest", help="SELECT the human-labelled oracle set")
    m.add_argument("--out", default="ht_manifest.csv")
    m.add_argument("--dive-id", type=int, action="append", help="restrict to dives")
    m.add_argument("--per-dive", type=int, default=0, help="cap rows per dive")
    m.add_argument("--limit", type=int, default=0)
    m.add_argument(
        "--checksums-out",
        help="also write the bare checksum list, one per line, for fetching JPEGs",
    )
    m.set_defaults(func=build_manifest)

    p = sub.add_parser("predict", help="download JPEGs, segment, gate on the laser")
    p.add_argument("--manifest", default="ht_manifest.csv")
    p.add_argument("--out", default="ht_results.csv")
    p.add_argument("--cache-dir", default="ht_jpeg_cache")
    p.add_argument("--no-cache", action="store_true", help="do not keep JPEGs on disk")
    p.add_argument(
        "--raw-root",
        default="",
        help=(
            "NAS root that `raw_path` is relative to. When set, an image with "
            "no JPEG in Garage is rectified locally from its .ORF instead of "
            "being skipped."
        ),
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument(
        "--resume",
        action="store_true",
        help="append to --out, skipping image_ids it already contains",
    )
    p.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--labels-bucket", default=DEFAULT_LABELS_BUCKET)
    p.add_argument("--labels-prefix", default=DEFAULT_LABELS_PREFIX)
    p.set_defaults(func=predict)

    v = sub.add_parser("visualize", help="render keypoint overlays for eyeballing")
    v.add_argument("--results", default="ht_results.csv")
    v.add_argument("--out-dir", default="ht_overlays")
    v.add_argument("--cache-dir", default="ht_jpeg_cache")
    v.add_argument("--no-cache", action="store_true")
    v.add_argument("--limit", type=int, default=12)
    v.add_argument("--status", default="predicted")
    v.add_argument("--worst", action="store_true", help="worst head error first")
    v.add_argument("--raw-root", default="")
    v.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT)
    v.add_argument("--region", default=DEFAULT_REGION)
    v.add_argument("--labels-bucket", default=DEFAULT_LABELS_BUCKET)
    v.add_argument("--labels-prefix", default=DEFAULT_LABELS_PREFIX)
    v.set_defaults(func=visualize)

    r = sub.add_parser("report", help="summarize a results CSV")
    r.add_argument("--results", default="ht_results.csv")
    r.set_defaults(func=report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
