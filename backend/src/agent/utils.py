import json
import logging
import os
import re

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def parse_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from LLM response: %s", text[:200])
        return {}


def get_llm(config, temperature: float = 0.0):
    """Create a ChatOpenAI instance from Configuration.

    Args:
        config: Configuration instance with llm_model, llm_base_url, llm_api_key.
        temperature: LLM temperature. Default 0.0.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.llm_model,
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        temperature=temperature,
        http_async_client=httpx.AsyncClient(proxy=None),
    )
