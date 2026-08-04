"""Fish controller for the FishSense API."""

import logging
from typing import List

from fastapi import Depends, HTTPException
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.dive import Dive
from fishsense_api.models.fish import Fish
from fishsense_api.models.image import Image
from fishsense_api.models.measurement import Measurement
from fishsense_api.models.species import Species
from fishsense_api.server import app

logger = logging.getLogger(__name__)


@app.get("/api/v1/fish/")
async def get_fish_list(
    session: AsyncSession = Depends(get_async_session),
) -> list[Fish]:
    """Retrieve all fish."""
    logger.debug("Retrieving all fish")
    query = select(Fish)

    return (await session.exec(query)).all()


@app.get("/api/v1/fish/{fish_id}")
async def get_fish(
    fish_id: int, session: AsyncSession = Depends(get_async_session)
) -> Fish | None:
    """Retrieve a fish by its ID."""
    logger.debug("Retrieving fish with id=%d", fish_id)
    query = select(Fish).where(Fish.id == fish_id)

    fish = (await session.exec(query)).first()
    if fish is None:
        logger.warning("Fish with id=%d not found", fish_id)
        raise HTTPException(status_code=404, detail="Fish not found")
    return fish


@app.get("/api/v1/fish/by-name/{name}")
async def get_fish_by_name(
    name: str, session: AsyncSession = Depends(get_async_session)
) -> Fish | None:
    """Retrieve a fish by its `name` natural key (physical fish models).

    404s when absent, matching `get_species_by_scientific_name`. Real fish carry
    `name=None` and are not reachable here — only named models are.
    """
    logger.debug("Retrieving fish with name=%s", name)
    query = select(Fish).where(Fish.name == name)

    fish = (await session.exec(query)).first()
    if fish is None:
        logger.warning("Fish with name=%s not found", name)
        raise HTTPException(status_code=404, detail="Fish not found")
    return fish


@app.post("/api/v1/fish", status_code=201)
async def post_fish(
    fish: Fish,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update a fish, upserting on the `name` natural key.

    `session.merge` keys on the primary key alone, so a body with `id=None`
    always INSERTs — which would hit `uq_fish_name` the second time the same
    model is measured. Resolving a named row to its existing id first turns the
    merge into an UPDATE (mirrors `post_measurement`). Real fish (`name=None`)
    skip the lookup and always insert — multiple NULLs are allowed under the
    unique constraint, so per-cluster real-fish identity is unaffected.
    """
    logger.debug("Creating a new fish")

    if fish.id is None and fish.name is not None:
        existing = (
            await session.exec(select(Fish).where(Fish.name == fish.name))
        ).first()
        if existing is not None:
            fish.id = existing.id

    fish = await session.merge(fish)
    await session.flush()

    fish_id = fish.id

    return fish_id


@app.get("/api/v1/dives/{dive_id}/measurements")
async def get_measurements_for_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> List[Measurement]:
    """Retrieve all measurements for a given dive ID.

    Stage 14 reads this once per dive to skip images it has already
    measured, which is what makes a re-run on a partially-measured dive
    safe.
    """
    logger.debug("Retrieving measurements for dive with id=%d", dive_id)
    query = (
        select(Measurement)
        .join_from(Measurement, Image, Measurement.image_id == Image.id)
        .join_from(Image, Dive, Image.dive_id == Dive.id)
        .where(Dive.id == dive_id)
    )

    measurements = (await session.exec(query)).all()
    if not measurements:
        logger.warning("Measurements for dive with id=%d not found", dive_id)
        raise HTTPException(status_code=404, detail="Measurements not found")
    return measurements


@app.post("/api/v1/fish/{fish_id}/measurements", status_code=201)
async def post_measurement(
    fish_id: int,
    measurement: Measurement,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update the measurement for a specific fish.

    Upserts on `(image_id, fish_id)`. `session.merge` keys on the primary
    key alone, so a body with `id=None` always INSERTed — which is how a
    re-run of stage 14 duplicated measurements on already-measured
    images. Resolving the natural key to an existing row id first turns
    the merge into an UPDATE.
    """
    logger.debug("Creating a new measurement for fish with id=%d", fish_id)
    measurement.fish_id = fish_id

    if measurement.id is None and measurement.image_id is not None:
        existing = (
            await session.exec(
                select(Measurement)
                .where(Measurement.image_id == measurement.image_id)
                .where(Measurement.fish_id == fish_id)
            )
        ).first()
        if existing is not None:
            measurement.id = existing.id

    measurement = await session.merge(measurement)
    await session.flush()

    measurement_id = measurement.id

    return measurement_id


@app.delete("/api/v1/fish/{fish_id}/measurements/{image_id}", status_code=204)
async def delete_measurement(
    fish_id: int,
    image_id: int,
    session: AsyncSession = Depends(get_async_session),
) -> None:
    """Delete the measurement binding `image_id` to `fish_id`.

    Exists so stage 14 can invalidate a measurement whose fish identity went
    stale — a species relabel (e.g. `Fish Model, Snook` -> `Fish Model,
    Grouper`) leaves the row bound to the old model's Fish, and re-measuring
    cannot fix it: `post_measurement` upserts on `(image_id, fish_id)`, so the
    corrected binding would be ADDED alongside and the image double-counted.

    Keyed on the same natural key as the upsert, not on `image_id` alone, so a
    frame holding two fish only loses the binding named here. Idempotent: a
    missing row is already the desired state, so no 404 — that keeps activity
    retries safe.
    """
    logger.debug(
        "Deleting measurement for image_id=%d fish_id=%d", image_id, fish_id
    )
    existing = (
        await session.exec(
            select(Measurement)
            .where(Measurement.image_id == image_id)
            .where(Measurement.fish_id == fish_id)
        )
    ).first()
    if existing is None:
        logger.debug(
            "No measurement for image_id=%d fish_id=%d; nothing to delete",
            image_id,
            fish_id,
        )
        return
    await session.delete(existing)
    await session.flush()


@app.get("/api/v1/fish/species/{scientific_name}")
async def get_species_by_scientific_name(
    scientific_name: str, session: AsyncSession = Depends(get_async_session)
) -> Species | None:
    """Retrieve a species by its scientific name."""
    logger.debug("Retrieving species with scientific_name=%s", scientific_name)
    query = select(Species).where(Species.scientific_name == scientific_name)

    species = (await session.exec(query)).first()
    if species is None:
        logger.warning("Species with scientific_name=%s not found", scientific_name)
        raise HTTPException(status_code=404, detail="Species not found")
    return species


@app.post("/api/v1/fish/species", status_code=201)
async def post_species(
    species: Species,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create a new species."""
    logger.debug("Creating a new species")
    species = await session.merge(species)
    await session.flush()

    species_id = species.id

    return species_id
