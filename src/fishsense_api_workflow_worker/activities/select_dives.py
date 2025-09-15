from typing import List

from temporalio import activity

from fishsense_api_workflow_worker.database import Database
from fishsense_api_workflow_worker.models.image import Image
from fishsense_api_workflow_worker.models.priority import Priority


@activity.defn
async def select_dives(database_url: str) -> List[Image]:
    """Select dives that need processing and mark them as 'in progress'."""
    log = activity.logger

    log.info("Selecting dives that need processing...")

    database = Database(database_url)

    dives = [
        d
        for d in await database.select_dives_to_process()
        if d.priority == Priority.HIGH
    ]

    if len(dives) > 0:
        dive = dives[0]

        return (dive, await database.select_images_by_dive_id(dive.id))

    return None, None
