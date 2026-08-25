"""Ingest one dive folder from the NAS into `Dive` + `Image` rows.

On-demand, no schedule. **One request means one dive** — the frames are the
`.ORF` files directly inside the named folder, not a recursive walk. That is
precedent, not simplification: the retired spider crawler assigned
`dive = image.parent`, so every one of the ~479 dive rows in prod is exactly one
directory, and recursing would merge dives that are distinct rows today.

```
temporal workflow start \\
    --task-queue fishsense_api_queue \\
    --type IngestDiveWorkflow \\
    --workflow-id ingest-<folder> \\
    --input '{"dive_path": "2024.06.20.REEF/082929_FishModels_FSL07",
              "self_calibrates": true, "priority": "HIGH"}'

temporal workflow query --workflow-id ingest-<folder> --type progress
```

Set `"dry_run": true` to stop after preflight, having written nothing.

**The shape is a two-phase commit, because there is no transaction spanning
these activities.** The dive is created at `priority=LOW` — which every hourly
cohort ignores — frames are registered in batches, and only then is it promoted
to the requested priority. Priority is the commit flag. A crash anywhere in the
middle leaves a dive and some images that no pipeline stage will touch, and
re-running is safe: `dives.post` upserts on `path`, the scan skips frames already
registered, and `finalize` refuses to promote an incomplete set.

Batching exists so a failure costs one batch of downloads rather than the whole
dive; the batch size is deliberately modest because each frame is ~14.5 MB and
the NAS is shared with the hourly staging activities doing real pipeline work.
"""

from datetime import timedelta
from typing import List

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.activities.finalize_dive_activity import (
        INCOMPLETE_INGEST_TYPE,
        IngestTotals,
    )
    from fishsense_api_workflow_worker.activities.list_dive_folder_activity import (
        DiveFolderListing,
    )
    from fishsense_api_workflow_worker.activities.scan_and_register_images_activity import (  # noqa: E501  pylint: disable=line-too-long
        BatchResult,
    )
    from fishsense_shared.ingest_contracts import (
        IngestDiveRequest,
        IngestPreflight,
        IngestProgress,
        IngestReport,
    )

# Frames per scan activity. Each is a whole-file download (~14.5 MB), so 25 is
# roughly 360 MB of work — small enough that a failure is cheap to redo, large
# enough not to pay activity overhead per frame.
BATCH_SIZE = 25

# A dive's frames are read serially inside the activity, so a batch can take a
# while; the heartbeat is what distinguishes slow from wedged.
_SCAN_TIMEOUT = timedelta(hours=2)

_NAS_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=30),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=5,
    # A missing file cannot be fixed by waiting.
    non_retryable_error_types=["NasFileNotFound"],
)


@workflow.defn
class IngestDiveWorkflow:
    """List, preflight, create, scan, finalize — one dive."""

    def __init__(self) -> None:
        self._progress = IngestProgress()

    @workflow.query
    def progress(self) -> IngestProgress:
        """Live counts. A large dive is hours of downloading, so the portal (and
        an operator watching by hand) needs to see movement without waiting for
        the return value."""
        return self._progress

    @workflow.run
    async def run(self, request: IngestDiveRequest) -> IngestReport:
        self._progress.state = "listing"
        listing: DiveFolderListing = await workflow.execute_activity(
            "list_dive_folder_activity",
            args=(request,),
            result_type=DiveFolderListing,
            schedule_to_close_timeout=timedelta(minutes=15),
            retry_policy=_NAS_RETRY,
        )

        self._progress.state = "preflight"
        self._progress.total = len(listing.files)
        preflight: IngestPreflight = await workflow.execute_activity(
            "preflight_ingest_activity",
            args=(request, listing),
            result_type=IngestPreflight,
            # Ranged 1 MB reads, serially, over every frame.
            schedule_to_close_timeout=timedelta(hours=2),
            heartbeat_timeout=timedelta(minutes=10),
            retry_policy=_NAS_RETRY,
        )

        if request.dry_run or preflight.errors:
            # Errors and a dry run leave by the same door: nothing has been
            # written either way, and the report IS the deliverable.
            self._progress.state = "rejected" if preflight.errors else "dry-run"
            return IngestReport(
                dive_path=request.dive_path,
                total=len(listing.files),
                committed=False,
                preflight=preflight,
            )

        self._progress.state = "creating"
        dive_id: int = await workflow.execute_activity(
            "create_dive_activity",
            args=(request, preflight),
            result_type=int,
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        self._progress.dive_id = dive_id

        self._progress.state = "scanning"
        totals = IngestTotals(total=len(preflight.images))
        paths: List[str] = [image.path for image in preflight.images]

        for start in range(0, len(paths), BATCH_SIZE):
            batch = paths[start : start + BATCH_SIZE]
            self._progress.current_path = batch[0]
            result: BatchResult = await workflow.execute_activity(
                "scan_and_register_images_activity",
                args=(dive_id, batch, preflight.resolved_camera_id),
                result_type=BatchResult,
                schedule_to_close_timeout=_SCAN_TIMEOUT,
                heartbeat_timeout=timedelta(minutes=15),
                retry_policy=_NAS_RETRY,
            )
            totals.registered += result.registered
            totals.skipped_existing += result.skipped_existing
            totals.rejected.extend(result.rejected)
            if result.max_taken_datetime is not None and (
                totals.max_taken_datetime is None
                or result.max_taken_datetime > totals.max_taken_datetime
            ):
                totals.max_taken_datetime = result.max_taken_datetime

            self._progress.scanned += len(batch)
            self._progress.registered = totals.registered
            self._progress.skipped_existing = totals.skipped_existing
            self._progress.rejected = len(totals.rejected)

        self._progress.state = "finalizing"
        self._progress.current_path = None
        report: IngestReport = await workflow.execute_activity(
            "finalize_dive_activity",
            args=(dive_id, request, totals),
            result_type=IngestReport,
            schedule_to_close_timeout=timedelta(minutes=15),
            retry_policy=RetryPolicy(
                maximum_attempts=3,
                # An incomplete set is a data problem: retrying re-reads the
                # same bytes and reaches the same conclusion.
                non_retryable_error_types=[INCOMPLETE_INGEST_TYPE],
            ),
        )

        self._progress.state = "done"
        report.preflight = preflight
        return report
