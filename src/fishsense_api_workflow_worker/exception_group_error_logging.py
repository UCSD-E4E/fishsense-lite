"""Context manager to report errors in asyncio TaskGroups."""


class ExceptionGroupErrorLogging:
    """Async context manager to report errors in asyncio TaskGroups.
    
    This context manager should wrap asyncio.TaskGroup() to properly catch
    and log ExceptionGroup errors that occur when tasks fail.
    """

    def __init__(self, activity_logger):
        self.activity_logger = activity_logger

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None and isinstance(exc, ExceptionGroup):
            # Log the overall ExceptionGroup error
            self.activity_logger.error(
                f"ExceptionGroup with {len(exc.exceptions)} sub-exception(s) occurred in TaskGroup"
            )

            # Log each individual exception with details
            for i, sub_exc in enumerate(exc.exceptions, 1):
                self.activity_logger.error(
                    f"Sub-exception {i}/{len(exc.exceptions)}: {type(sub_exc).__name__}: {sub_exc}",
                    exc_info=(type(sub_exc), sub_exc, sub_exc.__traceback__)
                )
        elif exc:
            # Log any other exception type
            self.activity_logger.error(
                f"Error in TaskGroup: {type(exc).__name__}: {exc}",
                exc_info=(exc_type, exc, tb)
            )
        return False  # Propagate exception if any
