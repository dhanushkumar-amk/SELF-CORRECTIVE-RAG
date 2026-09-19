"""
Structured logging setup for the application.

Provides a consistent logging configuration across all modules.
"""

import logging
import sys


def get_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    """Return a configured logger with structured formatting.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level as int or string (e.g., "DEBUG", "INFO"). Defaults to ``logging.INFO``.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if isinstance(level, str):
        resolved_level = getattr(logging, level.upper(), logging.INFO)
    else:
        resolved_level = level

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(resolved_level)
    return logger
