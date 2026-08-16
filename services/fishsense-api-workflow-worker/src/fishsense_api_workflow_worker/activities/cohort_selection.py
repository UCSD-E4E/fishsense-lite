"""Shared body for the nine per-stage cohort selectors.

Every `select_next_high_priority_dive_for_*_activity` does the same three
things: open an SDK client, call one `dives.select_next_for_*()` endpoint, and
log which dive came back (or that the cohort is empty). Only the endpoint and
the stage name differ. They were nine copies of that body, and the copies
drifted — the laser selector's docstring *and its operator-facing log line*
described a "no LaserExtrinsics row" cohort for months after the endpoint had
moved to "image with no non-sentinel LaserLabel". A log line that lies about
what was queried is expensive at 3am.

The activity functions themselves stay one-per-module and keep their explicit
`@activity.defn` name. That is deliberate: Temporal resolves activities by
string, and workflows name them as strings, so a factory that generated them
would make `select_next_high_priority_dive_for_laser_preprocessing_activity`
ungreppable. Each module keeps its cohort docstring too — the cohort
*definition* is the genuinely per-stage knowledge, and it belongs next to the
endpoint it names. What's shared here is only the plumbing.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client

__all__ = ["select_next_dive"]


async def select_next_dive(
    stage: str,
    select: Callable[[object], Awaitable[int | None]],
) -> int | None:
    """Run one cohort selector and log the outcome.

    `stage` is the human-readable stage name used in both log lines, so the
    "nothing to do" and "picked a dive" messages can't describe different
    things. `select` receives the open SDK client and calls the one endpoint
    for this stage — passed as a callable rather than a method name so the
    actual `fs.dives.select_next_for_*()` call stays written out, and
    greppable, in the calling module.

    Returns the next dive_id, or None when the cohort is empty (the parent
    workflow then ends the firing without dispatching anything).
    """
    async with get_fs_client() as fs:
        dive_id = await select(fs)

    if dive_id is None:
        activity.logger.info("no HIGH-priority dives needing %s", stage)
    else:
        activity.logger.info(
            "next HIGH-priority dive needing %s: dive_id=%d", stage, dive_id
        )
    return dive_id
