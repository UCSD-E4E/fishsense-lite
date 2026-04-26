"""Fish controller for the FishSense API."""

import logging

from fastapi import Depends, HTTPException
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.fish import Fish
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


@app.post("/api/v1/fish", status_code=201)
async def post_fish(
    fish: Fish,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create a new fish."""
    logger.debug("Creating a new fish")
    fish = await session.merge(fish)
    await session.flush()

    fish_id = fish.id

    return fish_id


@app.post("/api/v1/fish/{fish_id}/measurements", status_code=201)
async def post_measurement(
    fish_id: int,
    measurement: Measurement,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create a new measurement for a specific fish."""
    logger.debug("Creating a new measurement for fish with id=%d", fish_id)
    measurement.fish_id = fish_id
    measurement = await session.merge(measurement)
    await session.flush()

    measurement_id = measurement.id

    return measurement_id


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
