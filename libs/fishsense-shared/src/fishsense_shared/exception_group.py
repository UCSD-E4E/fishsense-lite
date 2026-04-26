"""Async context manager that logs every sub-exception in an asyncio.TaskGroup.

Wrap ``asyncio.TaskGroup()`` with this to make individual task failures visible
(by default the TaskGroup raises a single ExceptionGroup whose contents only
appear in the bare exception traceback).
"""


class ExceptionGroupErrorLogging:
    """Async context manager to report errors in asyncio TaskGroups."""

    def __init__(self, activity_logger):
        self.activity_logger = activity_logger

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
        return False
