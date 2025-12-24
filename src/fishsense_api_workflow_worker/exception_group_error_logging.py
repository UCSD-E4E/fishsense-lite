"""Context manager to report errors in asyncio TaskGroups."""


class ExceptionGroupErrorLogging:
    """Context manager to report errors in asyncio TaskGroups."""

    def __init__(self, activity_logger):
        self.activity_logger = activity_logger

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc:
            self.activity_logger.error(f"Error in TaskGroup: {exc}")
        return False  # Propagate exception if any
