"""Plan — and, only against a reviewed plan, apply — laser-label revivals.

On-demand, never scheduled, started by `RemediateLaserSupersedesParentWorkflow`
(api-worker), which wakes this light queue first. Returns the report.

Dry run unless `apply` is set AND the plan recomputed here has the digest of
the report a human reviewed. A plan with nothing left to revive is a clean
no-op rather than a mismatch, which is what makes re-applying safe. The apply
activity re-checks each dive against current state on top of this.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from fishsense_shared.laser_remediation import (
        RemediateLaserSupersedesInput,
        RemediationDiveRequest,
        revival_digest,
    )

# The light pod runs 8 activities at a time; more in flight only queues.
_BATCH = 8

_PLAN_TIMEOUTS = {
    "schedule_to_close_timeout": timedelta(minutes=15),
    "start_to_close_timeout": timedelta(minutes=10),
    "heartbeat_timeout": timedelta(minutes=1),
}


@workflow.defn
class RemediateLaserSupersedesWorkflow:
    # pylint: disable=too-few-public-methods
    """Report per dive; write only a reviewed plan."""

    @workflow.run
    async def run(self, request: RemediateLaserSupersedesInput) -> dict:
        excluded_dives = set(request.excluded_dive_ids)
        requests = [
            RemediationDiveRequest(
                dive_id=dive_id,
                excluded_label_ids=list(request.excluded_label_ids),
                dive_excluded=dive_id in excluded_dives,
            )
            for dive_id in sorted(set(request.dive_ids))
        ]

        rows: list[dict] = []
        for start in range(0, len(requests), _BATCH):
            rows += await asyncio.gather(
                *(
                    workflow.execute_activity(
                        "plan_laser_supersede_remediation_activity",
                        args=(r,),
                        **_PLAN_TIMEOUTS,
                    )
                    for r in requests[start : start + _BATCH]
                )
            )

        digest = revival_digest((row["dive_id"], row["revive_ids"]) for row in rows)
        report = {
            "mode": "dry_run",
            "plan_sha256": digest,
            "excluded_dive_ids": sorted(excluded_dives),
            "excluded_label_ids": sorted(set(request.excluded_label_ids)),
            "totals": {
                "dives": len(rows),
                "positives": sum(row["positives"] for row in rows),
                "superseded_now": sum(row["superseded_now"] for row in rows),
                "superseded_after": sum(row["superseded_after"] for row in rows),
                "to_revive": sum(len(row["revive_ids"]) for row in rows),
                "reflection_suspects": sum(
                    1 for row in rows if row["reflection_suspect"] is not None
                ),
            },
            "dives": rows,
            "applied": None,
        }
        if not request.apply:
            return report

        to_apply = [row for row in rows if row["revive_ids"]]
        if not to_apply:
            report["mode"] = "apply_noop"
            report["applied"] = {}
            return report
        if request.expected_plan_sha256 != digest:
            raise ApplicationError(
                f"refusing to apply: the plan now has digest {digest}, the "
                f"reviewed report has {request.expected_plan_sha256}. Re-run "
                "the dry run and review it again. Nothing was written.",
                type="RemediationPlanMismatch",
                non_retryable=True,
            )

        applied: dict[str, int] = {}
        for start in range(0, len(to_apply), _BATCH):
            batch = to_apply[start : start + _BATCH]
            written = await asyncio.gather(
                *(
                    workflow.execute_activity(
                        "apply_laser_supersede_remediation_activity",
                        args=(
                            RemediationDiveRequest(
                                dive_id=row["dive_id"],
                                excluded_label_ids=list(request.excluded_label_ids),
                                dive_excluded=row["dive_id"] in excluded_dives,
                                revive_ids=list(row["revive_ids"]),
                            ),
                        ),
                        **_PLAN_TIMEOUTS,
                    )
                    for row in batch
                )
            )
            applied.update({str(row["dive_id"]): n for row, n in zip(batch, written)})
        report["mode"] = "apply"
        report["applied"] = applied
        return report
