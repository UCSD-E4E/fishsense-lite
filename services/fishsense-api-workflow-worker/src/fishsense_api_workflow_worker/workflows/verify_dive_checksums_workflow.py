"""Workflow to re-hash one migrated dive against the NAS.

On-demand, no schedule, read-only. Exists so "do we trust the migrated data?"
can be answered by an operator on demand rather than by a one-off script:

  * it runs inside the slot, where the NAS credentials already live, so nobody
    has to copy them anywhere to ask the question;
  * whole-file downloads over FileStation's fragile backend get Temporal's
    bounded jittered retry and a heartbeat that resumes mid-dive instead of
    re-pulling gigabytes;
  * the report is durable and visible in the Temporal UI, so a finding can be
    pointed at later rather than living in someone's terminal scrollback.

```
temporal workflow start \
    --task-queue fishsense_api_queue \
    --type VerifyDiveChecksumsWorkflow \
    --workflow-id verify-checksums-<dive_id> \
    --input <dive_id> --input 25
```

The second argument samples: verification pulls whole files (~15 MB each, the
one thing preflight's ranged read avoids), and answering "does the convention
hold" does not need all ~500 frames of a dive. Pass `null` to check every one.
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from fishsense_shared.ingest_contracts import VerifyChecksumsReport


@workflow.defn
class VerifyDiveChecksumsWorkflow:
    # pylint: disable=too-few-public-methods
    """Compare stored `checksum` / `taken_datetime` against the files on NAS."""

    @workflow.run
    async def run(
        self, dive_id: int, limit: int | None = None
    ) -> VerifyChecksumsReport:
        """Verify `dive_id`, optionally sampling the first `limit` frames."""
        return await workflow.execute_activity(
            "verify_dive_checksums_activity",
            args=(dive_id, limit),
            # Generous: a full dive is ~500 whole-file downloads over a backend
            # that is deliberately driven serially.
            schedule_to_close_timeout=timedelta(hours=6),
            # One frame at a time, so a gap this long means the transfer is
            # wedged rather than slow.
            heartbeat_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=30),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(minutes=5),
                # Bounded: this is diagnostic, not load-bearing. If the NAS is
                # having a bad day the answer is to ask again later, not to
                # keep hammering a fragile shared download backend.
                maximum_attempts=5,
            ),
        )
