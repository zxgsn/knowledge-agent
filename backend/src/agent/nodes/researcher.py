"""Research nodes: query generation, web search, reflection."""

from __future__ import annotations

import json
import os
import time
import urllib.request

import httpx
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.logger import get_logger, log_context
from agent.prompts import (
    HYDE_PROMPT,
    QUERY_REWRITE_PROMPT,
    QUERY_WRITER_PROMPT,
    REFLECTION_PROMPT,
    WEB_SEARCHER_PROMPT,
)
from agent.heartbeat import heartbeat
from agent.retry import with_retry
from agent.state import AgentState
from agent.utils import get_llm, parse_json

logger = get_logger(__name__)

# Transient errors worth retrying for network/LLM calls
_TRANSIENT_ERRORS = (
    ConnectionError,
    TimeoutError,
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.HTTPStatusError,
)


# #region debug-point B:report-helper
def _debug_report(hypothesis_id: str, location: str, msg: str, data: dict | None = None) -> None:
    payload = {
        "sessionId": "frontend-network-error",
        "runId": "pre-fix",
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
# #endregion


@with_retry(
    max_retries=3,
    base_delay=1.0,
    max_delay=15.0,
    retryable_exceptions=_TRANSIENT_ERRORS,
)
async def _tavily_search(query: str, api_key: str) -> list[dict]:
    """Call Tavily search API (async) with automatic retry."""
    logger.debug("Tavily search request: %s", query[:100])
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": 5,
                "search_depth": "advanced",
            },
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        logger.info(
            "Tavily search returned %d results for query: %s",
            len(results),
            query[:80],
        )
        return results


async def generate_query(state: AgentState, config: RunnableConfig) -> dict:
    """Generate search queries from the user's message."""
    try:
        return await _generate_query_impl(state, config)
    except Exception as exc:
        logger.error("generate_query failed catastrophically: %s", exc, exc_info=True)
        # Fallback: use raw user message as query
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                return {"search_query": [msg.content], "research_loop_count": 0}
        return {"search_query": ["general research"], "research_loop_count": 0}


async def _generate_query_impl(state: AgentState, config: RunnableConfig) -> dict:
    """Implementation of generate_query, wrapped with top-level error handler."""
    with log_context():
        dispatch_custom_event(
            "progress",
            {"stage": "generate_query", "detail": "Generating search queries..."},
            config=config,
        )
        configurable = Configuration.from_runnable_config(config)
        llm = get_llm(configurable, temperature=0.7)

        research_topic = _get_research_topic(state["messages"])
        prompt = QUERY_WRITER_PROMPT.format(
            number_queries=configurable.number_of_initial_queries,
            research_topic=research_topic,
        )

        try:
            # #region debug-point D:generate-query
            started = time.perf_counter()
            _debug_report("D", "backend/src/agent/nodes/researcher.py:generate_query:before", "[DEBUG] generate_query LLM call starting", {
                "research_topic_length": len(research_topic),
            })
            response = await llm.ainvoke(prompt)
            _debug_report("D", "backend/src/agent/nodes/researcher.py:generate_query:after", "[DEBUG] generate_query LLM call completed", {
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "response_length": len(response.content or ""),
            })
            # #endregion
        except Exception as exc:
            _debug_report("B", "backend/src/agent/nodes/researcher.py:generate_query:error", "[DEBUG] generate_query LLM call failed", {
                "error_type": type(exc).__name__,
                "error": repr(exc),
            })
            logger.error("LLM call failed in generate_query: %s", exc)
            # Graceful fallback: use the raw research topic as the single query
            return {
                "search_query": [research_topic],
                "research_loop_count": 0,
            }

        parsed = parse_json(response.content)
        queries = parsed.get("queries", [research_topic])

        if not isinstance(queries, list) or not queries:
            logger.warning(
                "generate_query produced no valid queries, falling back to topic"
            )
            queries = [research_topic]

        logger.info("Generated %d search queries: %s", len(queries), [q[:60] for q in queries])

        return {
            "search_query": queries,
            "research_loop_count": 0,
        }


async def web_research(state: dict, config: RunnableConfig) -> dict:
    """Perform web search for a single query and summarize results."""
    try:
        return await _web_research_impl(state, config)
    except Exception as exc:
        logger.error("web_research failed catastrophically: %s", exc, exc_info=True)
        return {
            "web_research_result": [f"Web research failed: {exc}"],
            "sources_gathered": [],
        }


async def _web_research_impl(state: dict, config: RunnableConfig) -> dict:
    """Implementation of web_research, wrapped with top-level error handler."""
    with log_context():
        configurable = Configuration.from_runnable_config(config)
        llm = get_llm(configurable, temperature=0.3)

        query = state["search_query"]
        api_key = os.getenv("TAVILY_API_KEY", "")

        if not api_key:
            logger.warning("No TAVILY_API_KEY configured, skipping web search")
            return {
                "web_research_result": [f"No Tavily API key configured. Cannot search for: {query}"],
                "sources_gathered": [],
            }

        dispatch_custom_event(
            "progress",
            {"stage": "web_research", "detail": f"Searching: {query[:80]}..."},
            config=config,
        )

        async with heartbeat(config, "web_research", f"Researching: {query[:60]}...", interval=8):
            try:
                # #region debug-point D:web-search-call
                search_started = time.perf_counter()
                _debug_report("D", "backend/src/agent/nodes/researcher.py:web_research:search-before", "[DEBUG] tavily search starting", {
                    "query": query[:200],
                })
                results = await _tavily_search(query, api_key)
                _debug_report("D", "backend/src/agent/nodes/researcher.py:web_research:search-after", "[DEBUG] tavily search completed", {
                    "elapsed_ms": round((time.perf_counter() - search_started) * 1000, 2),
                    "result_count": len(results),
                })
                # #endregion
            except Exception as exc:
                _debug_report("B", "backend/src/agent/nodes/researcher.py:web_research:search-error", "[DEBUG] tavily search failed", {
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                    "query": query[:200],
                })
                logger.error("Tavily search failed after retries for '%s': %s", query[:80], exc)
                results = []

            search_text = ""
            sources = []
            for r in results:
                search_text += f"- {r.get('title', '')}: {r.get('content', '')[:500]}\n  URL: {r.get('url', '')}\n"
                sources.append({"title": r.get("title", ""), "url": r.get("url", "")})

            dispatch_custom_event(
                "progress",
                {"stage": "web_research", "detail": "Summarizing search results..."},
                config=config,
            )

            try:
                prompt = WEB_SEARCHER_PROMPT.format(search_results=search_text)
                # #region debug-point D:web-summary-call
                summarize_started = time.perf_counter()
                _debug_report("D", "backend/src/agent/nodes/researcher.py:web_research:summary-before", "[DEBUG] web summary LLM call starting", {
                    "query": query[:200],
                    "search_text_length": len(search_text),
                })
                response = await llm.ainvoke(prompt)
                summary = response.content
                _debug_report("D", "backend/src/agent/nodes/researcher.py:web_research:summary-after", "[DEBUG] web summary LLM call completed", {
                    "elapsed_ms": round((time.perf_counter() - summarize_started) * 1000, 2),
                    "summary_length": len(summary or ""),
                })
                # #endregion
            except Exception as exc:
                _debug_report("B", "backend/src/agent/nodes/researcher.py:web_research:summary-error", "[DEBUG] web summary LLM call failed", {
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                    "query": query[:200],
                })
                logger.error("LLM summarization failed in web_research: %s", exc)
                summary = f"Search returned {len(results)} results but summarization failed. Raw results preserved."

        return {
            "web_research_result": [summary],
            "sources_gathered": sources,
        }


async def reflection(state: AgentState, config: RunnableConfig) -> dict:
    """Analyze gathered research and identify gaps."""
    try:
        return await _reflection_impl(state, config)
    except Exception as exc:
        logger.error("reflection failed catastrophically: %s", exc, exc_info=True)
        return {
            "is_sufficient": True,
            "knowledge_gap": "",
            "follow_up_queries": [],
            "research_loop_count": state.get("research_loop_count", 0) + 1,
        }


async def _reflection_impl(state: AgentState, config: RunnableConfig) -> dict:
    """Implementation of reflection, wrapped with top-level error handler."""
    with log_context():
        dispatch_custom_event(
            "progress",
            {"stage": "reflection", "detail": "Analyzing research completeness..."},
            config=config,
        )
        configurable = Configuration.from_runnable_config(config)
        llm = get_llm(configurable, temperature=0.3)

        research_topic = _get_research_topic(state["messages"])
        summaries = "\n\n---\n\n".join(state.get("web_research_result", []))

        prompt = REFLECTION_PROMPT.format(
            research_topic=research_topic,
            summaries=summaries,
        )

        async with heartbeat(config, "reflection", "Analyzing research...", interval=8):
            try:
                # #region debug-point D:reflection-call
                reflection_started = time.perf_counter()
                _debug_report("D", "backend/src/agent/nodes/researcher.py:reflection:before", "[DEBUG] reflection LLM call starting", {
                    "research_topic_length": len(research_topic),
                    "summaries_length": len(summaries),
                })
                response = await llm.ainvoke(prompt)
                _debug_report("D", "backend/src/agent/nodes/researcher.py:reflection:after", "[DEBUG] reflection LLM call completed", {
                    "elapsed_ms": round((time.perf_counter() - reflection_started) * 1000, 2),
                    "response_length": len(response.content or ""),
                })
                # #endregion
            except Exception as exc:
                _debug_report("B", "backend/src/agent/nodes/researcher.py:reflection:error", "[DEBUG] reflection LLM call failed", {
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                })
                logger.error("LLM call failed in reflection: %s", exc)
                # On failure, assume research is sufficient so the pipeline can complete
                return {
                    "is_sufficient": True,
                    "knowledge_gap": "",
                    "follow_up_queries": [],
                    "research_loop_count": state.get("research_loop_count", 0) + 1,
                }

        parsed = parse_json(response.content)

        is_sufficient = parsed.get("is_sufficient", True)
        follow_ups = parsed.get("follow_up_queries", [])

        logger.info(
            "Reflection: sufficient=%s, gaps='%s', follow_ups=%d",
            is_sufficient,
            str(parsed.get("knowledge_gap", ""))[:80],
            len(follow_ups) if isinstance(follow_ups, list) else 0,
        )

        return {
            "is_sufficient": is_sufficient,
            "knowledge_gap": parsed.get("knowledge_gap", ""),
            "follow_up_queries": follow_ups,
            "research_loop_count": state.get("research_loop_count", 0) + 1,
        }


def _get_research_topic(messages: list) -> str:
    """Extract the research topic from message history."""
    topic_parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            topic_parts.append(msg.content)
        elif isinstance(msg, AIMessage) and msg.content:
            topic_parts.append(msg.content[:200])
    return "\n".join(topic_parts) if topic_parts else "general research"


@with_retry(
    max_retries=2,
    base_delay=1.0,
    max_delay=10.0,
    retryable_exceptions=_TRANSIENT_ERRORS,
)
async def generate_hypothetical_document(query: str, config: RunnableConfig) -> str:
    """Generate a hypothetical document that would answer the query (HyDE)."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.hyde_enabled:
        return query

    llm = get_llm(configurable, temperature=0.3)
    prompt = HYDE_PROMPT.format(query=query)
    response = await llm.ainvoke(prompt)
    logger.debug("HyDE generated for query: %s", query[:60])
    return response.content


@with_retry(
    max_retries=2,
    base_delay=1.0,
    max_delay=10.0,
    retryable_exceptions=_TRANSIENT_ERRORS,
)
async def rewrite_query(
    messages: list, latest_message: str, config: RunnableConfig
) -> str:
    """Rewrite an ambiguous query using conversation history."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.memory_query_rewrite_enabled:
        return latest_message

    # Build history from recent messages
    history_parts = []
    for msg in messages[-6:]:
        if isinstance(msg, HumanMessage):
            history_parts.append(f"User: {msg.content[:200]}")
        elif isinstance(msg, AIMessage) and msg.content:
            history_parts.append(f"Assistant: {msg.content[:200]}")
    history = "\n".join(history_parts)

    if not history:
        return latest_message

    llm = get_llm(configurable, temperature=0.3)
    prompt = QUERY_REWRITE_PROMPT.format(history=history, latest_message=latest_message)
    response = await llm.ainvoke(prompt)
    logger.debug("Query rewritten: '%s' -> '%s'", latest_message[:60], response.content[:60])
    return response.content.strip()
