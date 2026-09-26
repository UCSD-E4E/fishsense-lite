"""Plan, and apply a reviewed plan for, reviving eroded laser labels.

The hourly validator used to re-fit only each dive's survivors and eroded
dives a run at a time (see `test_laser_validator_does_not_erode.py`). These two
activities undo that under review: `plan_...` says per dive which superseded
labels one full-population judgement keeps (the validator's own `judge_dive`,
so a revival is never re-superseded), and `apply_...` writes a plan a human
has seen.

Apply trusts nothing it is handed. It re-plans from current state and refuses,
writing nothing, if any requested id is not in that plan — an id that was
excluded, flagged, edited since the dry run, or never planned. Ids already live
are skipped, so re-applying is a no-op.
"""

from __future__ import annotations

import asyncio

from fishsense_api_sdk.models.superseded_reason import SupersededReason
from fishsense_shared.laser_remediation import RemediationDiveRequest
from temporalio import activity
from temporalio.exceptions import ApplicationError

from fishsense_data_processing_workflow_worker.activities.heartbeat import (
    HEARTBEAT_INTERVAL_SECONDS,
    heartbeat_pump,
)
from fishsense_data_processing_workflow_worker.activities.utils import get_fs_client
from fishsense_data_processing_workflow_worker.laser_label_validation.judgement import (
    calibration_image_ids,
)
from fishsense_data_processing_workflow_worker.laser_label_validation.remediation import (
    plan_dive,
)

__all__ = [
    "apply_laser_supersede_remediation_activity",
    "plan_laser_supersede_remediation_activity",
    "WRITE_CONCURRENCY",
]

# Same cap as the validator's supersede writes, for the same reasons.
WRITE_CONCURRENCY = 8


async def _plan(fs, request: RemediationDiveRequest):
    labels = (
        await fs.labels.get_laser_labels(request.dive_id, include_superseded=True) or []
    )
    calibration_ids = calibration_image_ids(
        await fs.labels.get_dive_slate_labels(request.dive_id) or []
    )
    plan = plan_dive(
        request.dive_id,
        labels,
        calibration_ids,
        excluded_label_ids=set(request.excluded_label_ids),
        dive_excluded=request.dive_excluded,
    )
    return plan, {label.id: label for label in labels}


@activity.defn
async def plan_laser_supersede_remediation_activity(
    request: RemediationDiveRequest,
) -> dict:
    """One dive's report row. Reads only."""
    async with heartbeat_pump(HEARTBEAT_INTERVAL_SECONDS), get_fs_client() as fs:
        plan, _ = await _plan(fs, request)
    return plan.to_dict()


@activity.defn
async def apply_laser_supersede_remediation_activity(
    request: RemediationDiveRequest,
) -> int:
    """Revive `request.revive_ids` on one dive. Returns how many it wrote."""
    async with heartbeat_pump(HEARTBEAT_INTERVAL_SECONDS), get_fs_client() as fs:
        plan, by_id = await _plan(fs, request)
        pending = [
            i
            for i in request.revive_ids
            if by_id.get(i) is None or by_id[i].superseded is not False
        ]
        unplanned = sorted(set(pending) - set(plan.revive_ids))
        if unplanned:
            raise ApplicationError(
                f"dive_id={request.dive_id}: refusing to revive {unplanned}; the "
                "current plan does not contain them (excluded, flagged by the "
                "fit, or changed since the dry run). Re-run the dry run and "
                "review it again. Nothing was written.",
                type="RemediationPlanMismatch",
                non_retryable=True,
            )

        sem = asyncio.Semaphore(WRITE_CONCURRENCY)

        async def _revive(label_id: int) -> None:
            label = by_id[label_id]
            label.superseded = False
            label.superseded_reason = SupersededReason.REMEDIATION
            async with sem:
                await fs.labels.put_laser_label(label.image_id, label)
            activity.logger.info(
                "dive_id=%d REVIVED laser_label_id=%d image_id=%s "
                "label_studio_task_id=%s label_studio_project_id=%s "
                "-> superseded=False reason=remediation",
                request.dive_id,
                label_id,
                label.image_id,
                label.label_studio_task_id,
                label.label_studio_project_id,
            )
            activity.heartbeat()

        await asyncio.gather(*(_revive(i) for i in pending))

    activity.logger.info(
        "dive_id=%d remediation revived %d laser labels (%d requested, %d already live)",
        request.dive_id,
        len(pending),
        len(request.revive_ids),
        len(request.revive_ids) - len(pending),
    )
    return len(pending)
