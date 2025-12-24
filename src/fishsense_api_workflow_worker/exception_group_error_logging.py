"""Context manager to report errors in asyncio TaskGroups."""


class ExceptionGroupErrorLogging:
    """Context manager to report errors in asyncio TaskGroups."""

    def __init__(self, activity_logger):
        self.activity_logger = activity_logger

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc:
            self.activity_logger.error(f"Error in TaskGroup: {exc}")
        return False  # Propagate exception if any
