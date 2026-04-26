"""Standard logging setup: file rotation + console, UTC timestamps, debug level."""

import logging
import logging.handlers
import time

from fishsense_shared.config import get_config_path, get_log_path

_LOG_FORMAT = "%(asctime)s.%(msecs)03dZ - %(name)s - %(levelname)s - %(message)s"
_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def configure_log_handler(handler: logging.Handler) -> None:
    """Apply the standard format/level to a logging handler."""
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))


def configure_logging(app_name: str, log_filename: str | None = None) -> None:
    """Configure root logging: TimedRotatingFileHandler + StreamHandler, both DEBUG.

    ``app_name`` is used to pick the log directory via platformdirs; the optional
    ``log_filename`` defaults to ``{app_name}.log``.
    """
    log_filename = log_filename or f"{app_name}.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    log_dest = get_log_path(app_name) / log_filename
    print(f'Logging to "{log_dest.as_posix()}"')

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dest, when="midnight", backupCount=5
    )
    configure_log_handler(file_handler)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    configure_log_handler(console_handler)
    root_logger.addHandler(console_handler)
    logging.Formatter.converter = time.gmtime

    logging.info("Log path: %s", get_log_path(app_name))
    logging.info("Config path: %s", get_config_path())
