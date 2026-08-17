"""Activity: list the canonical dives, for the migration audit sweep.

Thin wrapper over `dives.get_canonical()` so the sweep workflow stays a pure
orchestrator with no SDK import of its own. Returns ids rather than `Dive`
rows: the sweep only iterates them, and a list of ints keeps the workflow's
event history small over a run that visits several hundred dives.
"""

from __future__ import annotations

from typing import List

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client

__all__ = ["select_canonical_dive_ids_activity"]


@activity.defn
async def select_canonical_dive_ids_activity() -> List[int]:
    async with get_fs_client() as fs:
        dives = await fs.dives.get_canonical() or []

    # Ascending so a sweep's progress is legible against dive ids, and so an
    # interrupted run can be resumed with an explicit tail of the list.
    dive_ids = sorted(d.id for d in dives if d.id is not None)
    activity.logger.info("canonical dives selected count=%d", len(dive_ids))
    return dive_ids
