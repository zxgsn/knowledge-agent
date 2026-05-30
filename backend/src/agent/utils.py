"""Shared utilities for the knowledge agent."""

from __future__ import annotations

import json
import os
import re

import httpx
from dotenv import load_dotenv

from agent.logger import get_logger
from agent.retry import with_retry

load_dotenv()

logger = get_logger(__name__)

# Transient errors that are worth retrying for LLM / network calls
_TRANSIENT_ERRORS = (
    ConnectionError,
    TimeoutError,
    httpx.ConnectError,
    httpx.TimeoutException,
)


def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def parse_json(text: str) -> dict | list:
    """Extract JSON from LLM response, handling markdown code blocks.

    Returns an empty dict if the text cannot be parsed as JSON, with a
    structured warning logged for debugging.  May return a list if the
    LLM responded with a JSON array.
    """
    if not text or not text.strip():
        logger.warning("Empty text passed to parse_json")
        return {}

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)

    # Strip leading/trailing whitespace that LLMs sometimes add inside fences
    text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse JSON from LLM response (error=%s): %s",
            exc,
            text[:300],
        )
        return {}

    if not isinstance(result, dict):
        # Lists and other JSON types are valid — return as-is for callers
        # that expect them (e.g. parse_json("[1,2,3]")).
        return result

    return result


@with_retry(
    max_retries=2,
    base_delay=1.0,
    max_delay=15.0,
    retryable_exceptions=_TRANSIENT_ERRORS,
)
def _create_llm_client(
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    http_async_client: httpx.AsyncClient,
):
    """Create a ChatOpenAI instance with retry for transient connection errors."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        http_async_client=http_async_client,
    )


def get_llm(config, temperature: float = 0.0):
    """Create a ChatOpenAI instance from Configuration.

    The underlying client creation is wrapped with retry logic to handle
    transient connection failures during initialization.

    Args:
        config: Configuration instance with llm_model, llm_base_url, llm_api_key.
        temperature: LLM temperature. Default 0.0.
    """
    return _create_llm_client(
        model=config.llm_model,
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        temperature=temperature,
        http_async_client=httpx.AsyncClient(proxy=None),
    )
