"""Structured logging utilities for the knowledge agent.

Provides a ``get_logger`` factory that returns loggers with consistent
formatting.  Supports both human-readable (default) and JSON-structured
output (enabled via the ``LOG_FORMAT=json`` environment variable).

Usage::

    from agent.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Processing request", extra={"thread_id": "abc-123"})

    # Or use the context manager for automatic correlation IDs:
    with log_context(thread_id="abc-123"):
        logger.info("Doing work")
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Generator

# ---------------------------------------------------------------------------
# Context variable for correlation IDs
# ---------------------------------------------------------------------------
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")

_LOG_FORMAT = os.getenv("LOG_FORMAT", "").lower()

_INITIALIZED = False


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach correlation ID if present
        corr = _correlation_id.get("")
        if corr:
            log_entry["correlation_id"] = corr

        # Forward any extra fields the caller attached
        for key in ("thread_id", "query", "result_count", "cache_hits",
                     "cache_misses", "cache_hit_rate", "attempt", "error"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class _HumanFormatter(logging.Formatter):
    """Coloured, human-friendly log format for terminal output."""

    _LEVEL_COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[35m",  # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._LEVEL_COLORS.get(record.levelname, "")
        corr = _correlation_id.get("")
        corr_part = f" [{corr[:8]}]" if corr else ""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        return (
            f"{color}{timestamp} {record.levelname:<8}{self._RESET} "
            f"{record.name}{corr_part}: {record.getMessage()}"
        )


def _init_root() -> None:
    """One-time root logger configuration."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    root = logging.getLogger("agent")
    if root.handlers:
        return  # Already configured externally

    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)

    if _LOG_FORMAT == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_HumanFormatter())

    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with consistent formatting.

    The logger is a child of the ``agent`` namespace so all agent logs
    share the same handler and format.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance ready for use.
    """
    _init_root()
    return logging.getLogger(name)


@contextmanager
def log_context(**kwargs: Any) -> Generator[None, None, None]:
    """Context manager that injects a correlation ID into log records.

    If no ``correlation_id`` is provided, one is generated automatically.

    Usage::

        with log_context(thread_id="abc-123"):
            logger.info("Doing work")  # will include the correlation ID

    All log records emitted inside the context will have the correlation ID
    attached (via ``_correlation_id`` context variable).
    """
    cid = kwargs.pop("correlation_id", None) or str(uuid.uuid4())[:12]
    token = _correlation_id.set(cid)
    try:
        yield
    finally:
        _correlation_id.reset(token)
