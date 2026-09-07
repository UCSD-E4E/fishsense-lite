"""Shared upsert for the three model-prediction tables.

`LaserPrediction`, `HeadTailPrediction` and `SlatePrediction` are written by
their respective predict stages with identical semantics: one row per image,
upserted on `image_id`, stamped server-side. Their PUT handlers were
byte-identical apart from the model name — which is exactly the duplication
`duplicate-code` cannot see, because `LaserPrediction.image_id == image_id` and
`SlatePrediction.image_id == image_id` are different strings. CLAUDE.md names
this shape and the four `put_*_label` handlers that had it; a green pylint says
nothing about it.

Both behaviours here are load-bearing, and each was a bug before it was a rule:

**Resolve the natural key before merging.** `merge` on `id=None` always
INSERTs, which violates the `uq_*_prediction_image` constraint the moment a
stage re-runs. That is the duplicate-key 500 all four `put_*_label` handlers
shipped with.

**Stamp `created_at` server-side, on every write.** Predictions upsert in
place, so a re-prediction overwrites the coordinates a labeler was actually
shown; the timestamp is the only thing that can separate "this row reached a
labeler" from "this row replaced what did". A client-supplied value is ignored
rather than honoured, because a replayed payload would otherwise date new
coordinates to an old run. Re-stamping is intended: the surviving row *is* the
prediction in force, so the field means "when this x/y was produced".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Type, TypeVar

from fastapi.encoders import jsonable_encoder
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

T = TypeVar("T")


async def upsert_prediction(
    session: AsyncSession, model: Type[T], image_id: int, prediction: T
) -> int:
    """Create or update `model`'s prediction row for `image_id`; return its id."""
    prediction = model.model_validate(jsonable_encoder(prediction))
    prediction.image_id = image_id

    if prediction.id is None:
        existing = (
            await session.exec(select(model).where(model.image_id == image_id))
        ).first()
        if existing is not None:
            # Resolve the natural key so `merge` updates in place.
            prediction.id = existing.id

    # After the id resolution, so it survives the merge rather than being
    # overwritten by the loaded row's value.
    prediction.created_at = datetime.now(timezone.utc)

    prediction = await session.merge(prediction)
    await session.flush()

    return prediction.id
