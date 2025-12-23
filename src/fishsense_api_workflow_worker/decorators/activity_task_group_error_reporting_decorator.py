"""Decorator to log exceptions from asyncio.TaskGroup in Temporal activities."""

from typing import Callable

from temporalio import activity


def activity_task_group_error_reporting(func: Callable) -> Callable:
    """Decorator to log exceptions from asyncio.TaskGroup in Temporal activities."""

    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ExceptionGroup as eg:
            for e in eg.exceptions:
                activity.logger.error(f"Task group exception: {e}")
            raise eg

    return wrapper
