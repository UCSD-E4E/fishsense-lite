from typing import Iterable

from sqlmodel.ext.asyncio.session import AsyncSession
from temporalio import activity

from fishsense_api_workflow_worker.database import Database
from fishsense_api_workflow_worker.models.dive import Dive
from fishsense_api_workflow_worker.models.dive_frame_cluster import (
    DiveFrameCluster,
    DiveFrameClusterImageMapping,
)
from fishsense_api_workflow_worker.models.image import Image


@activity.defn
async def store_dive_clusters(
    dive: Dive, clusters: Iterable[Iterable[Image]], database_url: str
) -> None:
    """Store the grouped dive frames."""

    database = Database(database_url)

    async with AsyncSession(database.engine) as session:
        for cluster in clusters:
            cluster = list(cluster)  # Convert to list for multiple iterations
            cluster.sort(key=lambda img: img.taken_datetime)

            dive_frame_cluster = DiveFrameCluster(dive_id=dive.id)
            dive_frame_cluster = await session.merge(dive_frame_cluster)
            await session.flush()  # Ensure ID is populated

            mappings = []
            for image in cluster:
                mapping = DiveFrameClusterImageMapping(
                    dive_frame_cluster_id=dive_frame_cluster.id, image_id=image.id
                )
                mappings.append(mapping)

            session.add_all(mappings)

        await session.commit()
