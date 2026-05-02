"""Shared retry policies for parent-workflow activity calls.

Activities that are pure SDK round-trips (selectors, resolvers) should
fail fast on a 5xx instead of being retried until `schedule_to_close`.
The default retry shape on `execute_activity` is unlimited attempts
within the timeout window, which masks real bugs as latency: a
consistent 500 from the API takes the activity's full timeout (5+ min
for these) before surfacing.

Caps attempts at 2 so a single transient blip can self-heal but a real
bug shows up in seconds, and marks `HTTPStatusError` non-retryable
because the SDK's `httpx`-level `@retry(tries=3)` already absorbed any
transient status; if we still see one at this layer it's a real bug.

Used by the selector + resolver `execute_activity` calls in the parent
workflows (preprocess 0.1/2/5.1/9, calibration 13, measure-fish 14).
"""

from datetime import timedelta

from temporalio.common import RetryPolicy

SDK_FAIL_FAST_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_attempts=2,
    non_retryable_error_types=["HTTPStatusError"],
)
