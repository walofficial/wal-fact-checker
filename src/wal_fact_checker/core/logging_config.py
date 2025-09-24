# File: src/wal_fact_checker/core/logging_config.py
"""Logging configuration for the WAL Fact Checker application."""

from __future__ import annotations

import logging
import sys
from typing import Final

import structlog

# Log format based on ADK documentation best practices
LOG_FORMAT: Final[str] = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """Configure logging for the application based on ADK best practices."""
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[structlog.stdlib.ExtraAdder()],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            # structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(
                colors=True,
                pad_level=False,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ],
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
