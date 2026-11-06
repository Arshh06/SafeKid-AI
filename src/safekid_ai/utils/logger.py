"""Centralized logging utility for SafeKid AI."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "safekid_ai", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a consistent logger instance for SafeKid AI modules.

    Args:
        name: Name of the logger or module.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

