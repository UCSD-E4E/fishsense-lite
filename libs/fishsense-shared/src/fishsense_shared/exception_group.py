"""Async context manager that logs every sub-exception in an asyncio.TaskGroup.

Wrap ``asyncio.TaskGroup()`` with this to make individual task failures visible
(by default the TaskGroup raises a single ExceptionGroup whose contents only
appear in the bare exception traceback).

Pass ``suppress=True`` for fire-and-forget workflow blocks. Inside a Temporal
workflow, the bare ``BaseExceptionGroup`` raised by ``asyncio.TaskGroup`` is
not a ``temporalio.exceptions.FailureError`` subclass, so letting it propagate
classifies the failure as ``WORKFLOW_TASK_FAILED_CAUSE_WORKFLOW_WORKER_UNHANDLED_FAILURE``
and the workflow task retries forever instead of failing the run. The default
stays ``suppress=False`` so activities and worker startup keep their existing
propagating behavior.

``suppress=True`` only swallows ``Exception`` subclasses (which includes
``ExceptionGroup`` — the all-``Exception``-subclass flavor of the group).
Control-flow signals like ``asyncio.CancelledError``, ``KeyboardInterrupt``,
and ``SystemExit`` are ``BaseException`` subclasses and always propagate; a
``BaseExceptionGroup`` that contains any of those is likewise not an
``ExceptionGroup`` and propagates. Without this, swallowing cancellation
would break Temporal's workflow-cancel semantics.
"""


class ExceptionGroupErrorLogging:
    """Async context manager to report errors in asyncio TaskGroups."""

    def __init__(self, activity_logger, *, suppress: bool = False):
        self.activity_logger = activity_logger
        self.suppress = suppress

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None and isinstance(exc, BaseExceptionGroup):
            self.activity_logger.error(
                "ExceptionGroup with %d sub-exception(s) occurred in TaskGroup",
                len(exc.exceptions),
            )
            for i, sub_exc in enumerate(exc.exceptions, 1):
                self.activity_logger.error(
                    "Sub-exception %d/%d: %s: %s",
                    i,
                    len(exc.exceptions),
                    type(sub_exc).__name__,
                    sub_exc,
                    exc_info=(type(sub_exc), sub_exc, sub_exc.__traceback__),
                )
        elif exc:
            self.activity_logger.error(
                "Error in TaskGroup: %s: %s",
                type(exc).__name__,
                exc,
                exc_info=(exc_type, exc, tb),
            )
        return self.suppress and isinstance(exc, Exception)
