# Laser-calibration fingerprint — roadmap & validation plan

Living doc for the laser-calibration hardening + the line-fingerprint feature
family. Checkboxes track status. Started 2026-08-01.

## Background — the bugs this came out of

The 2026-07-31 slate dump surfaced two defects (fixed in #462) that left slate
labels in composite space and could mis-pair solvePnP points. Repairing the
104 labels and recomputing calibrations then exposed a latent `LaserExtrinsics`
bug, and the recovery raised a bigger question: how do we know when one dive's
calibration is safe to reuse for another?

## The core insight — the 2D laser line is the mount-state fingerprint

On a fixed camera the laser dots across a dive's frames are collinear in image
space (the projection of the fixed laser ray). That 2D line
`a*x + b*y + c = 0` (Hesse normal form) changes **only** when the cold-shoe
mount rotates, drifts (PLA thermal/creep), or is swapped.

Why the line is *sufficient*, not just necessary (the physical argument):
- The mount rotates in a cold shoe (so time is a poor proxy — it can move
  between same-day dives or stay fixed for months).
- PLA mounts deform with heat and there were **multiple undated PLA swaps**, so
  dates cannot identify which mount was in use.
- But every mount is the **same shape seated in the same cold shoe**, so the
  laser's realizable 3D positions are a tiny constrained family. Within it you
  can't land a *different* 3D ray on the *same* 2D projection — matching the
  line forces the same 3D geometry.

Therefore: **same camera + matching confident line ⇒ same calibration.** Fit
tightness (`residual_std` / `line_confidence`) doubles as a within-dive
stability signal (a mount deforming mid-dive smears the dots off a clean line).

The fingerprint comes from **laser labels, not slate labels**, so every dive
with confident laser labeling has one — not just slate-calibrated dives.

`(camera_id, line-fingerprint-cluster)` = **mount-state epoch**, and that single
grouping is the key for borrow, drift, swap-detection, and pooled calibration.

## Status

### Done
- [x] #462 — slate panel-offset fail-hard + multi-skip fix + remove "Slate
      upside down" (deployed: api-worker v1.46.5, data-worker v2.9.2).
- [x] Backfill 104 slate labels composite→photo space (verified).
- [x] Recompute 6/8 dives' `LaserExtrinsics` from corrected labels + re-measure
      (0 NaN). Dives 347 & 466 couldn't recompute (2 and 0 valid laser labels);
      their stale calibration + measurements were **deleted** so they drop out
      of `calibrated`/`measured` rather than feed confidently-wrong data.
- [x] #466 — `LaserExtrinsics` upsert on `dive_id` + non-null `created_at`
      (fixes the NULL-created_at 404 + duplicate-append; unique constraint +
      dedup/backfill migration). Merged.
- [x] #467 — persist per-dive `DiveLaserLine` fingerprint (model + PUT/GET +
      SDK + migration + validation-activity writeback). **In review.**

### Next (each depends on the prior)
- [ ] Merge + deploy #467 (endpoint must be live before fingerprints can write).
- [ ] Backfill `DiveLaserLine` for historical dives with laser labels.
- [ ] Early free check: do camera-3 dives **383 & 471** have matching lines AND
      agreeing independent calibrations? (both already self-calibrated.)
- [ ] Borrow-candidate finder: `GET /dives/{id}/calibration-candidates/` —
      hard-gated on (same `camera_id` + both fits confident + line agreement
      within tolerance), tightness-ranked, temporal only as advisory,
      **suggest-only** (human/portal calls `set_calibration_source`).
- [ ] Validation experiment (below) → lock the finder tolerance from results.
- [ ] Epoch grouping → drift analysis, mount-swap detection, pooled calibration.

## Validation experiment (gates trusting borrow in prod)

Prove "same line ⇒ same calibration" empirically before relying on it.

- **Positive control:** same-camera dives with matching fingerprints,
  independently calibrated (label slates → stage-13), must yield agreeing
  `LaserExtrinsics` — `laser_axis` angle small, `laser_position` L2 small.
- **Negative control (makes it rigorous):** same-camera dives with *different*
  fingerprints (across a suspected swap/drift) must yield *different*
  calibrations — proves the fingerprint **discriminates** mount states, not
  that the laser simply never moves.
- **Tolerance calibration:** the line-distance at which calibrations start
  disagreeing *is* the finder's θ_max / ρ_max. The experiment sets the
  threshold from data, not a guess.
- **Human-in-the-loop:** the slate labeling for chosen candidate dives is the
  one step not automatable; pick the dives that form the cleanest pos/neg
  controls and run them through labeling.

## Follow-on questions the fingerprint unlocks

- **Drift** — per-camera line params over time; translate a line shift into mm
  of length error via the triangulation. Within-mount = gentle wander, swap =
  step.
- **Mount-swap detection** — change-points in the per-camera series (step >
  local jitter). Honest limit: continuous PLA drift + multiple swaps can
  conflate a big thermal event with a swap; label steps "probable swap" for
  human confirmation.
- **Pooled calibration** — dives sharing a fingerprint co-fit one calibration
  (same gate as borrow). Rescues thin dives where a camera has enough *total*
  slate observations across an epoch. (347 has no camera-2 partner, so it stays
  stuck — capability pays off elsewhere.)

## Related follow-ups (not started)

- [ ] Slate predict activity (ML slate labeling from
      `UCSD-E4E/2026-07-31_slate_training`) — the original driver that makes
      recalibration routine and these features high-value.
