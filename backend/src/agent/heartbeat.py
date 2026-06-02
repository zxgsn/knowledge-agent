"""Heartbeat utility to keep SSE connections alive during long operations.

Provides an async context manager that spawns a background task to send
periodic progress events. This prevents browser/proxy timeouts when backend
nodes execute long-running operations without producing output events.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request
from contextlib import asynccontextmanager

from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig


def _debug_report(hypothesis_id: str, location: str, msg: str, data: dict | None = None) -> None:
    """Send debug event to debug server (if running)."""
    payload = {
        "sessionId": "frontend-network-error",
        "runId": "post-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "msg": msg,
        "data": data or {},
        "ts": int(time.time() * 1000),
    }
    debug_url = "http://127.0.0.1:7777/event"
    env_path = os.path.join(".dbg", "frontend-network-error.env")
    try:
        with open(env_path, encoding="utf-8") as env_file:
            for line in env_file:
                if line.startswith("DEBUG_SERVER_URL="):
                    debug_url = line.split("=", 1)[1].strip() or debug_url
                    break
    except Exception:
        pass
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                debug_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=2,
        ).read()
    except Exception:
        pass


@asynccontextmanager
async def heartbeat(
    config: RunnableConfig,
    stage: str,
    detail: str = "Processing...",
    interval: float = 10.0,
):
    """Send periodic progress events to keep SSE connection alive.

    Usage:
        async with heartbeat(config, "web_research", "Searching the web..."):
            # long-running operation
            result = await some_slow_function()

    Args:
        config: RunnableConfig from the LangGraph node
        stage: The stage name (e.g., "web_research", "respond")
        detail: Human-readable status message
        interval: Seconds between heartbeat events (default: 10)
    """
    heartbeat_count = 0

    async def _tick():
        nonlocal heartbeat_count
        while True:
            await asyncio.sleep(interval)
            heartbeat_count += 1
            try:
                dispatch_custom_event(
                    "progress",
                    {"stage": stage, "detail": f"{detail} (still working...)"},
                    config=config,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                pass  # Ignore dispatch errors during heartbeat
            try:
                _debug_report(
                    "D",
                    f"backend/src/agent/heartbeat.py:{stage}",
                    "[DEBUG] heartbeat tick",
                    {"stage": stage, "heartbeat_count": heartbeat_count},
                )
            except Exception:
                pass

    task = asyncio.create_task(_tick())
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        if heartbeat_count > 0:
            try:
                _debug_report(
                    "D",
                    f"backend/src/agent/heartbeat.py:{stage}:done",
                    "[DEBUG] heartbeat completed",
                    {
                        "stage": stage,
                        "total_heartbeats": heartbeat_count,
                        "elapsed_seconds": round(elapsed, 2),
                    },
                )
            except Exception:
                pass
