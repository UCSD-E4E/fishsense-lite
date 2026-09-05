"""Judge a dive's laser predictions and record which may skip human review.

Runs after the predict stage has persisted the dive's predictions. Fetches the
dive's WHOLE prediction set — not just the images the last run predicted —
because the gate's safety argument is consensus across the dive, and a line
fitted on a handful of newly-added frames is not that.

Lives on the data-worker for the reason every other math stage does: the
kernel is numpy and the SDK fetch stays inline beside it rather than being
split across a worker boundary for no gain.

Writes are minimal by design. A dive is re-judged whenever the predict parent
revisits it, and re-PUTting hundreds of unchanged rows every pass is pure load
on the API, so only genuine changes go back. The one thing it will always do
is *clear* a verdict that no longer holds: a re-predicted dive can lose the
consensus it had, and an `auto_accept=True` left standing from an earlier fit
would let a frame skip review on the strength of a line that no longer exists.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import List

import numpy as np
from fishsense_api_sdk.models.laser_prediction import LaserPrediction
from fishsense_shared import LASER_PREDICTOR_VERSION, LaserAutoAcceptSummary
from temporalio import activity

from fishsense_data_processing_workflow_worker.activities.heartbeat import (
    heartbeat_pump,
)
from fishsense_data_processing_workflow_worker.activities.utils import get_fs_client
from fishsense_data_processing_workflow_worker.laser_label_validation.auto_accept import (  # noqa: E501  pylint: disable=line-too-long
    AutoAcceptConfig,
    DiveIneligibleReason,
    FrameVerdict,
    evaluate_dive,
)

# Bound on concurrent verdict PUTs per dive, for the same reason the
# supersede path caps at 8: the SDK's timeout x retry ladder compounds
# linearly on sequential PUTs, and a large dive would otherwise walk into
# `start_to_close` on writes alone.
WRITE_CONCURRENCY = 8

__all__ = ["evaluate_laser_auto_accept_activity", "WRITE_CONCURRENCY"]


def _config_from_settings(source) -> AutoAcceptConfig:
    """Build the gate config from a settings object.

    Takes the source rather than reading the global so it can be exercised
    against a plain namespace — Dynaconf validates *every* validator on first
    attribute access, so a unit test that touched the real settings would have
    to plumb placeholders for the whole worker.
    """
    section = source.laser_auto_accept
    return AutoAcceptConfig(
        enabled=bool(section.enabled),
        min_predictions=int(section.min_predictions),
        min_inlier_fraction=float(section.min_inlier_fraction),
        max_perpendicular_px=float(section.max_perpendicular_px),
        max_along_line_z=float(section.max_along_line_z),
        audit_sample_rate=float(section.audit_sample_rate),
    )


def _resolve_config() -> AutoAcceptConfig:
    """The live gate config. Function-local import so importing this module
    does not trip Dynaconf's eager validation — same reason
    `object_store.open_client` keeps its `config` import local.
    """
    # pylint: disable=import-outside-toplevel
    from fishsense_data_processing_workflow_worker.config import settings

    return _config_from_settings(settings)


def _points(predictions: List[LaserPrediction]) -> np.ndarray:
    """(N, 2) of predicted dot positions, NaN where the detector abstained.

    NaN rather than dropping the row: the gate returns one verdict per input,
    and an abstention still needs recording (the frame needs a person) even
    though it takes no part in the fit.
    """
    return np.array(
        [
            (
                (float(p.x), float(p.y))
                if p.x is not None and p.y is not None
                else (np.nan, np.nan)
            )
            for p in predictions
        ],
        dtype=float,
    ).reshape(-1, 2)


def _changed(prediction: LaserPrediction, decision) -> bool:
    """Whether this row's stored verdict already matches the decision.

    Margins are compared loosely — they are recorded diagnostics, and a
    float round-trip through JSON should not manufacture a write.
    """
    if prediction.auto_accept != decision.auto_accept:
        return True
    if prediction.gate_verdict != decision.reason.value:
        return True
    for stored, computed in (
        (prediction.line_offset_px, decision.perpendicular_px),
        (prediction.line_position_z, decision.along_line_z),
    ):
        if (stored is None) != (computed is None):
            return True
        if stored is not None and abs(stored - computed) > 1e-6:
            return True
    return False


async def _refuse_dive(
    fs,
    dive_id: int,
    predictions: List[LaserPrediction],
    reason: DiveIneligibleReason,
    enabled: bool,
) -> LaserAutoAcceptSummary:
    """Mark every frame ineligible and clear any verdict that no longer holds.

    Used for a refusal decided *before* the fit, where there is no line and so
    no `perpendicular_px` / `along_line_z` to record. Clearing matters as much
    as refusing: a dive that was auto-accepted under an earlier run and has
    since been found stale must not leave `auto_accept=True` rows standing,
    which is the same reason the ordinary path rewrites verdicts that changed.
    """
    written = 0
    semaphore = asyncio.Semaphore(WRITE_CONCURRENCY)

    async def _write(prediction: LaserPrediction) -> None:
        nonlocal written
        async with semaphore:
            prediction.auto_accept = False
            prediction.gate_verdict = FrameVerdict.DIVE_INELIGIBLE.value
            prediction.line_offset_px = None
            prediction.line_position_z = None
            await fs.labels.put_laser_prediction(prediction.image_id, prediction)
            written += 1
            activity.heartbeat()

    stale_or_standing = [
        p
        for p in predictions
        if p.auto_accept
        or p.gate_verdict != FrameVerdict.DIVE_INELIGIBLE.value
        or p.line_offset_px is not None
    ]
    await asyncio.gather(*(_write(p) for p in stale_or_standing))

    return LaserAutoAcceptSummary(
        dive_id=dive_id,
        enabled=enabled,
        eligible=False,
        reason=reason.value,
        n_points=len(predictions),
        auto_accepted=0,
        verdicts={FrameVerdict.DIVE_INELIGIBLE.value: len(predictions)},
        written=written,
    )


@activity.defn
async def evaluate_laser_auto_accept_activity(dive_id: int) -> LaserAutoAcceptSummary:
    """Run the auto-accept gate over `dive_id` and persist the verdicts.

    Returns the per-dive summary. The caller logs it; the verdict histogram is
    the monitoring signal for the whole stage, so an eligible dive reports its
    numbers just as loudly as a refused one.
    """
    activity.logger.info("dive_id=%d auto-accept gate starting", dive_id)
    activity.heartbeat()

    async with heartbeat_pump(), get_fs_client() as fs:
        predictions = await fs.labels.get_laser_predictions(dive_id) or []
        activity.heartbeat()

        if not predictions:
            activity.logger.info("dive_id=%d has no laser predictions", dive_id)
            return LaserAutoAcceptSummary(dive_id=dive_id, eligible=False)

        config = _resolve_config()

        # **Refuse the whole dive if ANY prediction is not from the current
        # detector.** Decided before fitting, and deliberately at dive level.
        #
        # Stage v1 -- and every pre-versioning row, which carries NULL --
        # hardcoded the laser pre-annotation to "Red Laser" rather than reading
        # the dot's colour (`fishsense_shared.laser_predictor`). Auto-accepting
        # one writes a possibly-wrong colour into the corpus with NO human in
        # the loop, because skipping review is precisely what this gate does;
        # nothing downstream re-checks it. In prod this was reachable: 1,296
        # such rows sat in the drain's path at one dive an hour.
        #
        # Whole dive, not the individual rows, because the gate's decision IS a
        # dive-level consensus -- a line fitted across two detector behaviours
        # is not a meaningful fit, so the current-version frames could not be
        # judged against it honestly either. The dive returns on its own once
        # the predict cohort's stale-version path has re-predicted it and the
        # new rows have cleared these verdicts.
        #
        # The cohort selector applies the same rule so the hourly drain never
        # even selects such a dive. This second copy is not redundant: the
        # predict parent runs this gate inline, and an operator can run it by
        # hand, so the cohort guards neither of those paths.
        stale = [
            p for p in predictions if p.predictor_version != LASER_PREDICTOR_VERSION
        ]
        if stale:
            activity.logger.warning(
                "dive_id=%d refused: %d/%d predictions are not from detector "
                "v%d (versions present: %s); re-predict before judging",
                dive_id,
                len(stale),
                len(predictions),
                LASER_PREDICTOR_VERSION,
                sorted({str(p.predictor_version) for p in stale}),
            )
            return await _refuse_dive(
                fs,
                dive_id,
                predictions,
                DiveIneligibleReason.STALE_PREDICTOR,
                config.enabled,
            )

        gate, decisions = evaluate_dive(
            dive_id,
            [p.image_id for p in predictions],
            _points(predictions),
            config=config,
        )
        counts = Counter(d.reason.value for d in decisions)
        activity.logger.info(
            "dive_id=%d gate enabled=%s eligible=%s reason=%s n=%d "
            "inliers=%d (%.0f%%) line_confidence=%.1f verdicts=%s",
            dive_id,
            config.enabled,
            gate.eligible,
            gate.reason.value if gate.reason else None,
            gate.n_points,
            gate.inlier_count,
            100.0 * gate.inlier_fraction,
            gate.line_confidence,
            dict(counts),
        )
        activity.heartbeat()

        pending = [
            (prediction, decision)
            for prediction, decision in zip(predictions, decisions)
            if _changed(prediction, decision)
        ]
        semaphore = asyncio.Semaphore(WRITE_CONCURRENCY)

        async def _write(prediction: LaserPrediction, decision) -> None:
            async with semaphore:
                prediction.auto_accept = decision.auto_accept
                prediction.gate_verdict = decision.reason.value
                prediction.line_offset_px = decision.perpendicular_px
                prediction.line_position_z = decision.along_line_z
                await fs.labels.put_laser_prediction(prediction.image_id, prediction)
                activity.heartbeat()

        await asyncio.gather(*(_write(p, d) for p, d in pending))

        # Counted off the flag, not off the verdict histogram. With the gate
        # disabled those two disagree on purpose — the histogram says what the
        # fit would have cleared, the flag says what actually may skip a human
        # — and every caller wants the second. In particular the parent uses it
        # to decide whether to walk the dive's Label Studio tasks at all.
        accepted = sum(1 for decision in decisions if decision.auto_accept)
        activity.logger.info(
            "dive_id=%d auto-accept gate wrote %d/%d rows; %d frames may skip "
            "review (%d would have, gate enabled=%s)",
            dive_id,
            len(pending),
            len(predictions),
            accepted,
            counts.get("auto_accepted", 0),
            config.enabled,
        )
        return LaserAutoAcceptSummary(
            dive_id=dive_id,
            enabled=config.enabled,
            eligible=gate.eligible,
            reason=gate.reason.value if gate.reason else None,
            n_points=gate.n_points,
            inlier_count=gate.inlier_count,
            inlier_fraction=gate.inlier_fraction,
            line_confidence=gate.line_confidence,
            auto_accepted=accepted,
            verdicts=dict(counts),
            written=len(pending),
        )
