"""Research nodes: query generation, web search, reflection."""

from __future__ import annotations

import os

import httpx
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.prompts import (
    HYDE_PROMPT,
    QUERY_REWRITE_PROMPT,
    QUERY_WRITER_PROMPT,
    REFLECTION_PROMPT,
    WEB_SEARCHER_PROMPT,
)
from agent.state import AgentState
from agent.utils import get_llm, parse_json


async def _tavily_search(query: str, api_key: str) -> list[dict]:
    """Call Tavily search API (async)."""
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
        return resp.json().get("results", [])


async def generate_query(state: AgentState, config: RunnableConfig) -> dict:
    """Generate search queries from the user's message."""
    dispatch_custom_event("progress", {"stage": "generate_query", "detail": "Generating search queries..."}, config=config)
    configurable = Configuration.from_runnable_config(config)
    llm = get_llm(configurable, temperature=0.7)

    research_topic = _get_research_topic(state["messages"])
    prompt = QUERY_WRITER_PROMPT.format(
        number_queries=configurable.number_of_initial_queries,
        research_topic=research_topic,
    )

    response = await llm.ainvoke(prompt)
    parsed = parse_json(response.content)
    queries = parsed.get("queries", [research_topic])

    return {
        "search_query": queries,
        "research_loop_count": 0,
    }


async def web_research(state: dict, config: RunnableConfig) -> dict:
    """Perform web search for a single query and summarize results."""
    configurable = Configuration.from_runnable_config(config)
    llm = get_llm(configurable, temperature=0.3)

    query = state["search_query"]
    api_key = os.getenv("TAVILY_API_KEY", "")

    if not api_key:
        return {
            "web_research_result": [f"No Tavily API key configured. Cannot search for: {query}"],
            "sources_gathered": [],
        }

    dispatch_custom_event("progress", {"stage": "web_research", "detail": f"Searching: {query[:80]}..."}, config=config)
    results = await _tavily_search(query, api_key)

    search_text = ""
    sources = []
    for r in results:
        search_text += f"- {r.get('title', '')}: {r.get('content', '')[:500]}\n  URL: {r.get('url', '')}\n"
        sources.append({"title": r.get("title", ""), "url": r.get("url", "")})

    dispatch_custom_event("progress", {"stage": "web_research", "detail": "Summarizing search results..."}, config=config)
    prompt = WEB_SEARCHER_PROMPT.format(search_results=search_text)
    response = await llm.ainvoke(prompt)

    return {
        "web_research_result": [response.content],
        "sources_gathered": sources,
    }


async def reflection(state: AgentState, config: RunnableConfig) -> dict:
    """Analyze gathered research and identify gaps."""
    dispatch_custom_event("progress", {"stage": "reflection", "detail": "Analyzing research completeness..."}, config=config)
    configurable = Configuration.from_runnable_config(config)
    llm = get_llm(configurable, temperature=0.3)

    research_topic = _get_research_topic(state["messages"])
    summaries = "\n\n---\n\n".join(state.get("web_research_result", []))

    prompt = REFLECTION_PROMPT.format(
        research_topic=research_topic,
        summaries=summaries,
    )

    response = await llm.ainvoke(prompt)
    parsed = parse_json(response.content)

    return {
        "is_sufficient": parsed.get("is_sufficient", True),
        "knowledge_gap": parsed.get("knowledge_gap", ""),
        "follow_up_queries": parsed.get("follow_up_queries", []),
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


async def generate_hypothetical_document(query: str, config: RunnableConfig) -> str:
    """Generate a hypothetical document that would answer the query (HyDE)."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.hyde_enabled:
        return query

    llm = get_llm(configurable, temperature=0.3)
    prompt = HYDE_PROMPT.format(query=query)
    response = await llm.ainvoke(prompt)
    return response.content


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
    return response.content.strip()
