"""Image Controller for FishSense API."""

import asyncio
import logging
from typing import Dict, List

from fastapi import Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.data_source import DataSource
from fishsense_api.models.dive_frame_cluster import (
    DiveFrameCluster,
    DiveFrameClusterImageMapping,
    DiveFrameClusterJson,
)
from fishsense_api.models.image import Image
from fishsense_api.server import app

logger = logging.getLogger(__name__)


@app.get("/api/v1/images/{image_id}")
async def get_image(
    image_id: int, session: AsyncSession = Depends(get_async_session)
) -> Image | None:
    """Retrieve an image by its ID."""
    logger.debug("Retrieving image with id=%d", image_id)
    query = select(Image).where(Image.id == image_id)

    image = (await session.exec(query)).first()
    if image is None:
        logger.warning("Image with id=%d not found", image_id)
        raise HTTPException(status_code=404, detail="Image not found")
    return image


@app.get("/api/v1/images/checksum/{checksum}")
async def get_image_by_checksum(
    checksum: str, session: AsyncSession = Depends(get_async_session)
) -> Image | None:
    """Retrieve an image by its checksum."""
    logger.debug("Retrieving image with checksum=%s", checksum)
    query = select(Image).where(Image.checksum == checksum)

    image = (await session.exec(query)).first()
    if image is None:
        logger.warning("Image with checksum=%s not found", checksum)
        raise HTTPException(status_code=404, detail="Image not found")
    return image


@app.get("/api/v1/dives/{dive_id}/images/")
async def get_dive_images(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> List[Image] | None:
    """Retrieve all images associated with a specific dive ID."""
    logger.debug("Retrieving images for dive with id=%d", dive_id)
    query = select(Image).where(Image.dive_id == dive_id)

    images = (await session.exec(query)).all()
    if not images:
        logger.warning("Images for dive with id=%d not found", dive_id)
        raise HTTPException(status_code=404, detail="Images not found")
    return images


@app.get("/api/v1/dives/{dive_id}/images/clusters/{data_source}")
async def get_clusters(
    dive_id: int,
    data_source: DataSource,
    session: AsyncSession = Depends(get_async_session),
) -> List[DiveFrameClusterJson] | None:
    """Retrieve all image clusters associated with a specific dive ID."""
    logger.debug(
        "Retrieving image clusters for dive with id=%d and data_source=%s",
        dive_id,
        data_source,
    )
    query = select(DiveFrameCluster).where(DiveFrameCluster.dive_id == dive_id)

    clusters = (await session.exec(query)).all()
    cluster_mapping_query = (
        select(DiveFrameClusterImageMapping)
        .join(
            DiveFrameCluster,
            DiveFrameClusterImageMapping.dive_frame_cluster_id == DiveFrameCluster.id,
        )
        .where(DiveFrameCluster.data_source == data_source)
        .where(DiveFrameCluster.dive_id == dive_id)
    )
    cluster_mappings = (await session.exec(cluster_mapping_query)).all()

    cluster_mappings_dict: Dict[int, List[DiveFrameClusterImageMapping]] = {}
    for mappings in cluster_mappings:
        if mappings.dive_frame_cluster_id not in cluster_mappings_dict:
            cluster_mappings_dict[mappings.dive_frame_cluster_id] = []

        cluster_mappings_dict[mappings.dive_frame_cluster_id].append(mappings)

    return [
        DiveFrameClusterJson(
            id=c.id,
            image_ids=[m.image_id for m in cluster_mappings_dict[c.id]],
            data_source=c.data_source,
            updated_at=c.updated_at,
            dive_id=c.dive_id,
            fish_id=c.fish_id,
        )
        for c in clusters
    ]


@app.post("/api/v1/dives/{dive_id}/images/clusters/", status_code=201)
async def post_cluster(
    dive_id: int,
    dive_frame_cluster: DiveFrameClusterJson,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create a new image cluster for a specific dive ID."""
    logger.debug("Creating a new image cluster for dive with id=%d", dive_id)
    dive_frame_cluster = DiveFrameClusterJson.model_validate(
        jsonable_encoder(dive_frame_cluster)
    )
    images = (
        await session.exec(
            select(Image).where(
                Image.id.in_(dive_frame_cluster.image_ids)  # pylint: disable=no-member
            )
        )
    ).all()

    dive_frame_cluster = DiveFrameCluster(
        dive_id=dive_id,
        data_source=dive_frame_cluster.data_source,
        updated_at=dive_frame_cluster.updated_at,
        fish_id=dive_frame_cluster.fish_id,
    )
    dive_frame_cluster = await session.merge(dive_frame_cluster)
    await session.flush()  # Ensure ID is populated

    dive_frame_cluster_id = dive_frame_cluster.id  # Access ID to ensure it's loaded

    mappings = []
    for image in images:
        mapping = DiveFrameClusterImageMapping(
            dive_frame_cluster_id=dive_frame_cluster.id, image_id=image.id
        )
        mappings.append(mapping)

    session.add_all(mappings)

    return dive_frame_cluster_id


@app.put("/api/v1/dives/{dive_id}/images/clusters/{dive_frame_cluster_id}")
async def put_cluster(
    dive_id: int,
    dive_frame_cluster_id: int,
    dive_frame_cluster: DiveFrameClusterJson,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Update an existing image cluster for a specific dive ID."""
    logger.debug(
        "Updating image cluster with id=%d for dive with id=%d",
        dive_frame_cluster_id,
        dive_id,
    )
    dive_frame_cluster = DiveFrameClusterJson.model_validate(
        jsonable_encoder(dive_frame_cluster)
    )
    images = (
        await session.exec(
            select(Image).where(
                Image.id.in_(dive_frame_cluster.image_ids)  # pylint: disable=no-member
            )
        )
    ).all()

    dive_frame_cluster = DiveFrameCluster(
        id=dive_frame_cluster_id,
        dive_id=dive_id,
        data_source=dive_frame_cluster.data_source,
        updated_at=dive_frame_cluster.updated_at,
        fish_id=dive_frame_cluster.fish_id,
    )
    dive_frame_cluster = await session.merge(dive_frame_cluster)
    await session.flush()  # Ensure ID is populated

    # Clear existing mappings
    mappings_to_delete = await session.exec(
        select(DiveFrameClusterImageMapping).where(
            DiveFrameClusterImageMapping.dive_frame_cluster_id == dive_frame_cluster.id
        )
    )
    await asyncio.gather(
        *[session.delete(mapping) for mapping in mappings_to_delete.all()]
    )

    mappings = []
    for image in images:
        mapping = DiveFrameClusterImageMapping(
            dive_frame_cluster_id=dive_frame_cluster.id, image_id=image.id
        )
        mappings.append(mapping)

    session.add_all(mappings)

    return dive_frame_cluster.id
