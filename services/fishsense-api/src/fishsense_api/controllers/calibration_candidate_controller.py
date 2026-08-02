"""Calibration-borrow candidate finder.

Given a dive, returns other dives whose laser-line fingerprint matches it, so
their `LaserExtrinsics` can be borrowed. Same camera + matching confident line
⇒ same laser geometry (the mount fingerprint), so the calibration transfers.
Suggest-only — the caller picks and calls `set_calibration_source`.
"""

import logging
import math
from datetime import datetime
from typing import List

from fastapi import Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.dive import Dive
from fishsense_api.models.dive_laser_line import DiveLaserLine
from fishsense_api.models.laser_extrinsics import LaserExtrinsics
from fishsense_api.server import app

logger = logging.getLogger(__name__)

# Default line-match tolerance. Starting values pending the labeling validation
# experiment (pos/neg controls) that will lock them from data; overridable per
# request. min_confidence mirrors the data-worker's LINE_CONFIDENCE_THRESHOLD
# (kept as a literal to avoid importing the worker).
_DEFAULT_MAX_ANGLE_DEG = 1.0
_DEFAULT_MAX_OFFSET_PX = 30.0
_DEFAULT_MIN_CONFIDENCE = 5.0


class CalibrationCandidate(BaseModel):
    """A dive whose laser-line fingerprint matches the target's, so its
    `LaserExtrinsics` is a borrow candidate. Read-only response DTO."""

    dive_id: int
    name: str | None
    camera_id: int | None
    dive_datetime: datetime
    laser_extrinsics_id: int
    line_angle_deg: float
    line_offset_px: float
    line_confidence: float
    residual_std: float
    days_apart: float


def _canonical_line(a: float, b: float, c: float) -> tuple[float, float, float]:
    """Sign-normalize a Hesse line so `(a,b,c)` and `(-a,-b,-c)` compare equal
    (make the normal point into a fixed half-plane)."""
    if a < 0 or (a == 0.0 and b < 0):
        return -a, -b, -c
    return a, b, c


def _line_distance(line_a: DiveLaserLine, line_b: DiveLaserLine) -> tuple[float, float]:
    """(normal angle in degrees, aligned offset |Δc| in px) between two lines."""
    a1, b1, c1 = _canonical_line(line_a.a, line_a.b, line_a.c)
    a2, b2, c2 = _canonical_line(line_b.a, line_b.b, line_b.c)
    dot = max(-1.0, min(1.0, a1 * a2 + b1 * b2))
    return math.degrees(math.acos(dot)), abs(c1 - c2)


@app.get("/api/v1/dives/{dive_id}/calibration-candidates/")
async def get_calibration_candidates(
    dive_id: int,
    max_angle_deg: float = _DEFAULT_MAX_ANGLE_DEG,
    max_offset_px: float = _DEFAULT_MAX_OFFSET_PX,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    session: AsyncSession = Depends(get_async_session),
) -> List[CalibrationCandidate]:
    """Dives whose laser-line fingerprint matches this dive's, ranked as
    calibration-borrow candidates. Suggest-only.

    Hard gate: same `camera_id`, the candidate has its own `LaserExtrinsics`,
    both dives have a confident line fit, and the lines agree within
    (`max_angle_deg`, `max_offset_px`). Ranked by line closeness, then fit
    tightness, then temporal proximity (advisory only — a rotating/swappable
    mount makes time an unreliable gate; the line is the real signal). Returns
    [] when the dive has no confident fingerprint of its own to match against.
    """
    target = await session.get(Dive, dive_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Dive not found")
    if target.camera_id is None:
        return []
    target_line = (
        await session.exec(
            select(DiveLaserLine).where(DiveLaserLine.dive_id == dive_id)
        )
    ).first()
    if target_line is None or target_line.line_confidence < min_confidence:
        return []

    rows = (
        await session.exec(
            select(Dive, DiveLaserLine, LaserExtrinsics)
            .join(DiveLaserLine, DiveLaserLine.dive_id == Dive.id)
            .join(LaserExtrinsics, LaserExtrinsics.dive_id == Dive.id)
            .where(Dive.camera_id == target.camera_id)
            .where(Dive.id != dive_id)
            .where(DiveLaserLine.line_confidence >= min_confidence)
        )
    ).all()

    candidates: list[CalibrationCandidate] = []
    for dive, line, extrinsics in rows:
        angle_deg, offset_px = _line_distance(target_line, line)
        if angle_deg > max_angle_deg or offset_px > max_offset_px:
            continue
        days_apart = (
            abs((dive.dive_datetime - target.dive_datetime).total_seconds()) / 86400.0
        )
        candidates.append(
            CalibrationCandidate(
                dive_id=dive.id,
                name=dive.name,
                camera_id=dive.camera_id,
                dive_datetime=dive.dive_datetime,
                laser_extrinsics_id=extrinsics.id,
                line_angle_deg=angle_deg,
                line_offset_px=offset_px,
                line_confidence=line.line_confidence,
                residual_std=line.residual_std,
                days_apart=days_apart,
            )
        )

    candidates.sort(
        key=lambda c: (c.line_angle_deg, c.line_offset_px, c.residual_std, c.days_apart)
    )
    return candidates
