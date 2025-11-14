"""Minimal loguru-backed logging helpers for project scripts."""

import logging
import sys
from typing import Any

from loguru import logger


_LOG_FORMAT = "<level>[{time:HH:mm:ss}][{level:^7}] {message}</level>"


def heading(title: str) -> None:
    """Log a colored section heading with surrounding whitespace."""
    spacer()
    logger.opt(colors=True, raw=True).info("<cyan>=== {} ===</cyan>\n", title)


def spacer() -> None:
    """Insert a blank line between sections without prefixes."""
    logger.opt(raw=True).info("\n")


def setup_logging(level: str = "INFO", **kwargs: Any) -> None:
    """Configure loguru once and quiet noisy third-party loggers."""
    if getattr(setup_logging, "_configured", False):
        return

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=_LOG_FORMAT,
        colorize=True,
        **kwargs,
    )

    for noisy in ("httpx", "botocore", "boto3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    setup_logging._configured = True


__all__ = ["logger", "setup_logging", "heading", "spacer"]
