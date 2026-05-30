"""Retry decorator with exponential backoff and jitter.

Provides ``with_retry`` for both sync and async callables.  Transient
failures (connection errors, timeouts, rate-limits) are retried
transparently so callers do not need to change their code.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException], Any] | None = None,
) -> Callable:
    """Decorator that retries a function on failure with exponential backoff + jitter.

    Works with both sync and async functions.

    Args:
        max_retries: Maximum number of retry attempts (0 = no retries).
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Upper bound for the delay (prevents unbounded waits).
        retryable_exceptions: Tuple of exception types that trigger a retry.
            Non-matching exceptions propagate immediately.
        on_retry: Optional callback ``(attempt, exception)`` invoked before
            each retry sleep.  Useful for emitting custom metrics.

    Returns:
        The decorated function.  On success it returns the result; on final
        failure it re-raises the last exception.
    """

    def _compute_delay(attempt: int) -> float:
        """Exponential backoff with full jitter."""
        exp = min(base_delay * (2 ** attempt), max_delay)
        return random.uniform(0, exp)

    def decorator(fn: Callable) -> Callable:
        # ------------------------------------------------------------------ #
        # Async path
        # ------------------------------------------------------------------ #
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exc: BaseException | None = None
                for attempt in range(max_retries + 1):
                    try:
                        return await fn(*args, **kwargs)
                    except retryable_exceptions as exc:
                        last_exc = exc
                        if attempt >= max_retries:
                            logger.error(
                                "All %d retries exhausted for %s: %s",
                                max_retries,
                                fn.__qualname__,
                                exc,
                            )
                            raise
                        delay = _compute_delay(attempt)
                        logger.warning(
                            "Retry %d/%d for %s after %.2fs: %s",
                            attempt + 1,
                            max_retries,
                            fn.__qualname__,
                            delay,
                            exc,
                        )
                        if on_retry is not None:
                            on_retry(attempt + 1, exc)
                        await asyncio.sleep(delay)
                # Should be unreachable, but satisfy type checkers.
                raise last_exc  # type: ignore[misc]

            return async_wrapper

        # ------------------------------------------------------------------ #
        # Sync path
        # ------------------------------------------------------------------ #
        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt >= max_retries:
                        logger.error(
                            "All %d retries exhausted for %s: %s",
                            max_retries,
                            fn.__qualname__,
                            exc,
                        )
                        raise
                    delay = _compute_delay(attempt)
                    logger.warning(
                        "Retry %d/%d for %s after %.2fs: %s",
                        attempt + 1,
                        max_retries,
                        fn.__qualname__,
                        delay,
                        exc,
                    )
                    if on_retry is not None:
                        on_retry(attempt + 1, exc)
                    time.sleep(delay)
            # Should be unreachable, but satisfy type checkers.
            raise last_exc  # type: ignore[misc]

        return sync_wrapper

    return decorator
