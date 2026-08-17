"""Sweep: re-hash every canonical dive against the NAS and report.

Every dive in the database arrived through the migration, so "does the
recovered convention actually hold?" is a question about the whole corpus.
`VerifyDiveChecksumsWorkflow` answers it for one dive; this answers it for all
of them.

**This runs on the api-worker, inside the slot, on the NAS's own network.** It
is never driven from a workstation: ~930 GB does not belong on a WAN link, and
the NAS credentials live in the slot, not on anyone's laptop. Temporal is what
makes that possible — start the workflow from anywhere, the bytes never leave
the LAN.

**Cost is what shapes the rest.** 65,981 canonical images at ~14.5 MB each is
**~930 GB**. Transport is `synology-filestation` >=0.2.0, which prefers SMB and
falls back to FileStation, and paces itself either way (`throttle=True`,
`max_concurrency=4`, `min_interval_ms=150`, internal retry capped at 60s
backoff) — that release is what fixed the FileStation download 502s. So the
constraint is no longer "the transport will fall over"; it is that ~930 GB
shares one NAS with the hourly staging activities doing real pipeline work.
Three consequences, all deliberate:

* **`limit_per_dive` is a runtime argument.** Whether a dive came through a
  *different ingest path* is a per-dive property, so ~5 frames per dive settles
  it for ~20 GB — about 2% of the cost. Checking every frame answers the
  different question "is any individual row corrupt". One workflow serves both;
  pick at start time.
* **Dives are verified one at a time.** The client already paces individual
  transfers; this is the layer above it, and it exists so a multi-day audit
  never outranks the pipeline for the NAS. There is no concurrency knob,
  because the correct value is 1 and a knob invites raising it.
* **One dive's failure does not abandon the sweep.** A full run is days of
  transfer; losing it because dive 300 of 479 was briefly unreachable would be
  its own outage. The dive is recorded with an `error` — deliberately a
  separate field, so an unreachable dive can never read as clean.

Read-only throughout; see `verify_dive_checksums_activity` for the tripwire.

```
# sample: ~5 frames/dive, ~20 GB, answers the migration question
temporal workflow start --task-queue fishsense_api_queue \
    --type VerifyAllDivesChecksumsWorkflow \
    --workflow-id verify-sweep-sample --input 5

# full: every canonical frame, ~930 GB, days
temporal workflow start --task-queue fishsense_api_queue \
    --type VerifyAllDivesChecksumsWorkflow \
    --workflow-id verify-sweep-full --input null

# just the dives a sample flagged
... --input null --input '[64, 66, 412]'
```

Progress is queryable while it runs (`temporal workflow query --type progress`),
which matters when the run lasts days.
"""

from datetime import timedelta
from typing import List

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from fishsense_shared.ingest_contracts import (
        DiveVerificationSummary,
        VerifyAllDivesProgress,
        VerifyAllDivesReport,
        VerifyChecksumsReport,
    )

# One dive can be ~500 whole-file downloads driven serially. The per-dive
# activity has its own bounded policy; this is the outer ceiling.
_PER_DIVE_TIMEOUT = timedelta(hours=6)


@workflow.defn
class VerifyAllDivesChecksumsWorkflow:
    """Verify the migrated checksums of every canonical dive."""

    def __init__(self) -> None:
        self._progress = VerifyAllDivesProgress()

    @workflow.query
    def progress(self) -> VerifyAllDivesProgress:
        """Live counts. A full sweep runs for days, so "is it still going, and
        how far?" must be answerable without waiting for the return value."""
        return self._progress

    @workflow.run
    async def run(
        self,
        limit_per_dive: int | None = None,
        dive_ids: List[int] | None = None,
    ) -> VerifyAllDivesReport:
        if dive_ids is None:
            self._progress.state = "selecting"
            dive_ids = await workflow.execute_activity(
                "select_canonical_dive_ids_activity",
                schedule_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

        report = VerifyAllDivesReport(dives_requested=len(dive_ids))
        self._progress.state = "verifying"
        self._progress.total_dives = len(dive_ids)

        for dive_id in dive_ids:
            self._progress.current_dive_id = dive_id
            # The WHOLE per-dive body sits inside the try, not just the
            # activity call. An exception escaping a workflow body fails the
            # workflow *task*, which Temporal reschedules indefinitely — so a
            # bug here manifests as a sweep that hangs forever rather than one
            # that fails. Catching wide keeps a multi-day run recoverable.
            try:
                result = await workflow.execute_activity(
                    "verify_dive_checksums_activity",
                    args=(dive_id, limit_per_dive),
                    # Required: without it the payload decodes to a plain dict
                    # and every attribute access below raises.
                    result_type=VerifyChecksumsReport,
                    schedule_to_close_timeout=_PER_DIVE_TIMEOUT,
                    heartbeat_timeout=timedelta(minutes=10),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=30),
                        backoff_coefficient=2.0,
                        maximum_interval=timedelta(minutes=5),
                        maximum_attempts=5,
                    ),
                )
                summary = DiveVerificationSummary(
                    dive_id=result.dive_id,
                    checked=result.checked,
                    total_in_dive=result.total_in_dive,
                    checksum_matched=result.checksum_matched,
                    mismatches=result.mismatches,
                    timestamp_mismatches=result.timestamp_mismatches,
                    missing_on_nas=result.missing_on_nas,
                    no_stored_checksum=result.no_stored_checksum,
                )
            except Exception as exc:  # pylint: disable=broad-except
                # Any per-dive failure is data about the corpus, not a reason to
                # discard the sweep — and `error` keeps it distinguishable from
                # a clean result, so an unreachable dive never reads as verified.
                workflow.logger.warning(
                    "verification failed dive_id=%d: %s", dive_id, exc
                )
                report.dives.append(
                    DiveVerificationSummary(dive_id=dive_id, error=str(exc))
                )
                self._progress.dives_errored += 1
                self._progress.dives_with_findings += 1
                self._progress.dives_done += 1
                continue

            report.dives.append(summary)
            report.dives_verified += 1
            report.images_checked += summary.checked
            report.checksum_matched += summary.checksum_matched

            self._progress.dives_done += 1
            self._progress.images_checked += summary.checked
            self._progress.checksum_matched += summary.checksum_matched
            if not summary.is_clean:
                self._progress.dives_with_findings += 1

        self._progress.state = "done"
        self._progress.current_dive_id = None
        workflow.logger.info(
            "sweep complete dives=%d verified=%d images=%d matched=%d findings=%d",
            report.dives_requested,
            report.dives_verified,
            report.images_checked,
            report.checksum_matched,
            len(report.dives_with_findings),
        )
        return report
