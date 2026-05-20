"""Response generation node."""

from __future__ import annotations

import asyncio
import json

import psycopg
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pgvector import Vector
from pgvector.psycopg import register_vector

from agent.configuration import Configuration
from agent.memory.core_memory import CoreMemory
from agent.prompts import ANSWER_PROMPT, SYSTEM_PROMPT, get_current_date
from agent.state import AgentState
from agent.storage import get_db_url, get_embeddings


def _search_archival_sync(query: str, limit: int = 3) -> list[dict]:
    """Search archival memory synchronously."""
    embeddings = get_embeddings()
    query_embedding = Vector(embeddings.embed_query(query))

    conn = psycopg.connect(get_db_url())
    register_vector(conn)
    rows = conn.execute(
        """
        SELECT content, metadata,
               1 - (embedding <=> %s::vector) AS score
        FROM archival_memory
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, limit),
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        score = float(row[2])
        if score >= 0.4:
            meta = row[1] if isinstance(row[1], dict) else json.loads(row[1])
            results.append({"content": row[0], "source": meta.get("source", ""), "score": score})
    return results


async def respond(state: AgentState, config: RunnableConfig) -> dict:
    """Generate the final response to the user."""
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(
        model=configurable.llm_model,
        base_url=configurable.llm_base_url,
        api_key=configurable.llm_api_key,
        temperature=0.5,
    )

    core_memory = CoreMemory.from_dict(state.get("core_memory", {}))
    research_topic = _get_research_topic(state["messages"])
    summaries = "\n\n---\n\n".join(state.get("web_research_result", []))
    archival_context = _format_archival_results(state.get("archival_results", []))
    sources = state.get("sources", [])

    system = SYSTEM_PROMPT.format(
        current_date=get_current_date(),
        memory_blocks=core_memory.compile(),
    )

    if state.get("mode") == "research" and summaries:
        user_content = ANSWER_PROMPT.format(
            research_topic=research_topic,
            summaries=summaries,
            archival_context=archival_context,
        )
    elif state.get("mode") == "recall" and archival_context:
        user_content = (
            f"{research_topic}\n\n"
            f"[Relevant knowledge from your archival memory]\n{archival_context}"
        )
    else:
        user_content = research_topic

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    response = await llm.ainvoke(messages)

    if sources:
        source_text = "\n\n**Sources:**\n"
        for i, s in enumerate(sources[:10], 1):
            source_text += f"[{i}] [{s.get('title', 'Link')}]({s.get('url', '')})\n"
        response.content += source_text

    return {"messages": [AIMessage(content=response.content)]}


def _get_research_topic(messages: list) -> str:
    parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            parts.append(msg.content)
        elif isinstance(msg, AIMessage) and msg.content:
            parts.append(msg.content[:200])
    return "\n".join(parts) if parts else ""


def _format_archival_results(results: list) -> str:
    if not results:
        return ""
    parts = []
    for r in results:
        score = r.get("score", 0)
        if score >= 0.3:
            parts.append(f"- {r['content']} (relevance: {score:.2f})")
    return "\n\n".join(parts) if parts else ""
