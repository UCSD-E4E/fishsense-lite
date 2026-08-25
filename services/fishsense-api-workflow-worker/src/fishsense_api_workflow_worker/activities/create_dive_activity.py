"""Activity: create the dive row, always at `priority=LOW`.

Half of a two-phase commit. Ingest writes a dive twice — LOW here, then the
requested priority in `finalize_dive_activity` once every frame has landed —
because there is no transaction spanning the activities in between.

**LOW is not negotiable, whatever the request asked for.** Every hourly cohort
selects on HIGH, so a dive created at HIGH before its images exist would be
picked up mid-ingest and processed against a partial set: clustered on some of
its frames, populated into Label Studio missing others. Priority is the commit
flag, and this is the half that keeps it closed.

`dives.post` upserts on `path`, so a re-run finds the same row rather than
creating a second one — which is also why re-running an interrupted ingest is
safe.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_shared.ingest_contracts import IngestDiveRequest, IngestPreflight

__all__ = ["create_dive_activity"]


def leaf_name(path: str) -> str:
    """The folder's own name, which is what a dive is called by default.

    `Dive.name` feeds the per-dive Label Studio project title, so leaving it
    NULL gives labelers a project called just `#412 - Species Labeling`.
    """
    return path.rstrip("/").rsplit("/", 1)[-1]


@activity.defn
async def create_dive_activity(
    request: IngestDiveRequest, preflight: IngestPreflight
) -> int:
    from fishsense_api_sdk.models.dive import Dive
    from fishsense_api_sdk.models.priority import Priority

    # `dive_datetime` is NOT NULL and no frame has been hashed yet, so seed it
    # from preflight's headers. `finalize` replaces it with the scan's max,
    # which is authoritative because the scan read every frame rather than a
    # 1 MB prefix.
    stamps = [i.taken_datetime for i in preflight.images if i.taken_datetime]
    provisional = max(stamps) if stamps else None
    if provisional is None:
        raise ValueError(
            "preflight produced no usable timestamps; refusing to create a dive "
            "with a fabricated dive_datetime"
        )

    dive = Dive(
        id=None,
        name=request.dive_name or leaf_name(request.dive_path),
        path=request.dive_path,
        dive_datetime=provisional,
        # See the module docstring: LOW regardless of `request.priority`.
        priority=Priority.LOW,
        flip_dive_slate=request.flip_dive_slate,
        camera_id=preflight.resolved_camera_id,
        dive_slate_id=request.dive_slate_id,
        # NULL means "self-calibrates". Writing a link for a self-calibrating
        # dive would be a lie the resolver happens to ignore (own wins).
        calibration_dive_id=request.calibration_dive_id,
    )

    async with get_fs_client() as fs:
        dive_id = await fs.dives.post(dive)

    activity.logger.info(
        "created dive id=%d path=%s at LOW (commit flag closed)",
        dive_id,
        request.dive_path,
    )
    return dive_id
