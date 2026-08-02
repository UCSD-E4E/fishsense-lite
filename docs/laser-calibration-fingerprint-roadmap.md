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

## Reframe (2026-08-02): borrow is mis-tuned; the real prize is self-calibration

Empirical work on the 219 backfilled fingerprints + the 6 existing calibrations
changed the picture. Findings:

- **Measurement is brutally sensitive to the laser axis: ~14% length error per
  degree** (position is forgiving, ~0.25%/mm). So the finder's 1° line
  tolerance is 10-25× too loose — measurement-grade borrow needs the axis to
  agree to ~0.1° (⇒ line match ~0.05-0.1°, essentially near-identical
  fingerprints), not 1°.
- **Cross-calibration transfer test** (measure dive B's fish with dive A's
  calibration, compare to B's own): the one same-camera pair (383↔471, lines
  1.1° apart) disagreed **44-305%** on length, and some cross-camera pairs beat
  it. So a 1°-"matching" fingerprint does NOT yield a transferable calibration.
- **But the geometry decomposes cleanly.** Per camera, all laser lines converge
  to a common apex (cam10 to 1.8px; cam5/1/4 to ~16-22px) ⇒ **fixed laser
  origin**. The 6 calibrations confirm it: origin = −30.9±1.0mm x, −101±5.7mm y
  across ALL rigs, while the axis varies 2-6°. And the stored calibration's
  projected line matches the observed fingerprint to <0.1° for the good fits ⇒
  **the fingerprint faithfully encodes the beam axis.**
- Mechanism (Chris): the laser diode spins in its bore, and because the beam is
  off the cylinder's body axis by some ε, that spin cones the beam — the axis
  changes with the mount perfectly rigid. Matches the data (fixed origin +
  rotating direction).

**Reframed north star — self-calibration from the laser line.** The calibration
is ~1 rotational DOF about a fixed, once-known geometry. So instead of borrowing
a neighbor's (noisy) calibration, fit each rig's laser model **once** (fixed 3D
origin + beam-body offset ε + body axis / the cone) from a handful of
well-labeled slate dives, then **every future dive self-calibrates from its own
laser line** — no per-dive slate labeling, no borrow-matching. This subsumes
borrow and largely removes the per-dive slate bottleneck (the slate detector
still produces the calibrations the rig model is fit from).

Open caveat: forward consistency (calibration → line) is confirmed tight, but
the inverse (line → 3D axis) has 1 residual DOF — the line fixes the beam's
plane, not its tilt within it (the depth-sensitive part). The per-rig **cone**
fit resolves it, and needs several calibrated dives on one camera.

### The experiment (single-camera cone fit + self-cal validation)

Label slates on **camera 5** (33 dives, apex residual 16px, already has 1
calibration = dive 279) for these high-confidence dives spanning the beam-angle
range, then fit the rig model, hold some out, and self-calibrate them from their
line alone vs their slate-derived calibration (and ideally ground-truth length):

| dive | date | line θ | fit conf |
|---|---|---|---|
| 467 | 2024-12-10 | −20.3° | 5350 |
| 192 or 446 | 2023-09-26 / 2024-10-27 | −18.1° | 6942 / 26966 |
| 246 | 2023-10-19 | −17.1° | 14494 |
| 375 | 2024-06-25 | −16.4° | 15820 |
| 279 ★ | 2023-11-13 | −14.9° | (calibrated) |
| 215 | 2023-09-26 | −13.4° | 7555 |

Deliberately skip the extreme low-confidence outliers (458 at −37°, 425 at +6°)
— almost certainly bad fits.

### Finder tolerance

Default `max_angle_deg=1.0` is far too loose for measurement-grade borrow; tighten
toward ~0.1° (and re-run coverage — the "98 borrowable" figure was at 1° and will
shrink a lot). Kept overridable so the validation experiment can set it from data.

## Related follow-ups (not started)

- [ ] Slate predict activity (ML slate labeling from
      `UCSD-E4E/2026-07-31_slate_training`) — the original driver that makes
      recalibration routine and these features high-value. Also produces the
      calibrations the per-rig cone model is fit from.
