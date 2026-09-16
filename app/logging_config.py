"""Central logging configuration, configured once at process startup."""
from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog + stdlib logging with a single shared level.

    Safe to call multiple times; subsequent calls are no-ops beyond
    re-applying the level.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Logs go to stderr, never stdout: the MCP server speaks JSON-RPC over
    # stdout and any stray log line there corrupts the protocol stream.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to the given module name."""
    return structlog.get_logger(name)
