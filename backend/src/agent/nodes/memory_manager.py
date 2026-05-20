"""Memory-related nodes: intent routing, archival save, memory recall."""

from __future__ import annotations

import asyncio
import json
import re

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from agent.configuration import Configuration
from agent.prompts import ARCHIVAL_STORE_PROMPT, ROUTE_INTENT_PROMPT
from agent.state import AgentState


def _parse_json(text: str) -> dict:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


async def route_intent(state: AgentState, config: RunnableConfig) -> dict:
    """Classify user intent: chat, research, or memory_edit."""
    configurable = Configuration.from_runnable_config(config)

    # If mode is already set (e.g. from frontend), keep it
    if state.get("mode") in ("research", "memory_edit", "ingest"):
        return {}

    llm = ChatOpenAI(
        model=configurable.llm_model,
        base_url=configurable.llm_base_url,
        api_key=configurable.llm_api_key,
        temperature=0,
    )

    user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break

    if not user_msg:
        return {"mode": "chat"}

    prompt = ROUTE_INTENT_PROMPT.format(user_message=user_msg)
    response = await llm.ainvoke(prompt)
    parsed = _parse_json(response.content)
    mode = parsed.get("mode", "chat")
    if mode not in ("chat", "research", "memory_edit", "ingest", "recall"):
        mode = "chat"

    need_recall = parsed.get("need_recall", False)

    return {"mode": mode, "need_recall": need_recall}


def _sync_search_archival(query: str, limit: int = 5, alpha: float = 0.7) -> list[dict]:
    """Hybrid search: vector similarity + BM25 keyword match.

    Args:
        query: Search query text.
        limit: Max results.
        alpha: Weight for vector score (1-alpha for BM25). Default 0.7.
    """
    import psycopg
    from pgvector import Vector
    from pgvector.psycopg import register_vector

    from agent.storage import get_db_url, get_embeddings

    embeddings = get_embeddings()
    query_embedding = Vector(embeddings.embed_query(query))

    conn = psycopg.connect(get_db_url())
    register_vector(conn)
    rows = conn.execute(
        """
        SELECT content, metadata,
               %s * (1 - (embedding <=> %s::vector))
                 + (1 - %s) * ts_rank(content_tsv, plainto_tsquery('simple', %s))
               AS score
        FROM archival_memory
        WHERE content_tsv @@ plainto_tsquery('simple', %s)
           OR 1 - (embedding <=> %s::vector) > 0.2
        ORDER BY score DESC
        LIMIT %s
        """,
        (alpha, query_embedding, alpha, query, query, query_embedding, int(limit)),
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        score = float(row[2])
        if score >= 0.01:
            meta = row[1] if isinstance(row[1], dict) else json.loads(row[1])
            results.append({"content": row[0], "metadata": meta, "score": score})
    return results


def _sync_save_to_recall(role: str, content: str, thread_id: str = "default") -> None:
    """Save a message to the recall_memory table."""
    import uuid

    import psycopg
    from pgvector import Vector
    from pgvector.psycopg import register_vector

    from agent.storage import ensure_recall_table, get_db_url, get_embeddings

    ensure_recall_table()
    embeddings = get_embeddings()
    entry_id = str(uuid.uuid4())
    embedding = Vector(embeddings.embed_query(content))

    conn = psycopg.connect(get_db_url())
    register_vector(conn)
    conn.execute(
        "INSERT INTO recall_memory (id, thread_id, role, content, embedding) "
        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
        (entry_id, thread_id, role, content, embedding),
    )
    conn.commit()
    conn.close()


async def recall_memory(state: AgentState, config: RunnableConfig) -> dict:
    """Search archival memory for content referenced by the user."""
    user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break

    if not user_msg:
        return {"archival_results": []}

    # Save user message to recall memory
    try:
        await asyncio.to_thread(_sync_save_to_recall, "user", user_msg)
    except Exception as e:
        import sys
        print(f"[memory_manager] Failed to save to recall: {e}", file=sys.stderr)

    results = await asyncio.to_thread(_sync_search_archival, user_msg, 5)

    return {
        "archival_results": [
            {
                "content": r["content"],
                "source": r["metadata"].get("source", ""),
                "score": r["score"],
            }
            for r in results
        ]
    }


def _sync_put_to_archival(content: str, namespace: str, metadata: dict) -> str:
    """Synchronous archival storage (runs in thread)."""
    import uuid
    from datetime import datetime, timezone

    import psycopg
    from pgvector import Vector
    from pgvector.psycopg import register_vector

    from agent.storage import get_db_url, get_embeddings

    embeddings = get_embeddings()
    entry_id = str(uuid.uuid4())
    metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
    embedding = Vector(embeddings.embed_query(content))

    conn = psycopg.connect(get_db_url())
    register_vector(conn)
    conn.execute(
        "INSERT INTO archival_memory (id, namespace, content, metadata, embedding) "
        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
        "content=EXCLUDED.content, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding",
        (entry_id, namespace, content, json.dumps(metadata), embedding),
    )
    conn.commit()
    conn.close()
    return entry_id


def _sync_put_batch_to_archival(entries: list[dict], namespace: str) -> list[str]:
    """Synchronous batch archival storage (runs in thread)."""
    import uuid
    from datetime import datetime, timezone

    import psycopg
    from pgvector import Vector
    from pgvector.psycopg import register_vector

    from agent.storage import get_db_url, get_embeddings

    if not entries:
        return []

    embeddings = get_embeddings()
    embedding_vectors = embeddings.embed_documents([e["content"] for e in entries])
    now = datetime.now(timezone.utc).isoformat()
    ids: list[str] = []

    conn = psycopg.connect(get_db_url())
    register_vector(conn)
    for entry, emb in zip(entries, embedding_vectors):
        entry_id = str(uuid.uuid4())
        meta = entry.get("metadata", {})
        meta["timestamp"] = now
        ids.append(entry_id)
        conn.execute(
            "INSERT INTO archival_memory (id, namespace, content, metadata, embedding) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
            "content=EXCLUDED.content, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding",
            (entry_id, namespace, entry["content"], json.dumps(meta), Vector(emb)),
        )
    conn.commit()
    conn.close()
    return ids


async def save_to_archival(state: AgentState, config: RunnableConfig) -> dict:
    """Extract key findings and save them to archival memory (PostgreSQL)."""
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(
        model=configurable.llm_model,
        base_url=configurable.llm_base_url,
        api_key=configurable.llm_api_key,
        temperature=0.3,
    )

    research_topic = _get_research_topic(state["messages"])
    summaries = "\n\n---\n\n".join(state.get("web_research_result", []))

    if not summaries:
        return {}

    prompt = ARCHIVAL_STORE_PROMPT.format(
        research_topic=research_topic,
        summaries=summaries,
    )
    response = await llm.ainvoke(prompt)

    try:
        entry_id = await asyncio.to_thread(
            _sync_put_to_archival,
            response.content,
            "research",
            {"source": "research_summary", "topic": research_topic},
        )
    except Exception as e:
        return {
            "archival_results": [{
                "content": response.content,
                "source": "research_summary",
                "topic": research_topic,
                "error": str(e),
            }]
        }

    return {
        "archival_results": [{
            "id": entry_id,
            "content": response.content,
            "source": "research_summary",
            "topic": research_topic,
        }]
    }


def _get_research_topic(messages: list) -> str:
    parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            parts.append(msg.content)
    return "\n".join(parts) if parts else ""


async def ingest_document_node(state: AgentState, config: RunnableConfig) -> dict:
    """Ingest a document (URL or text) into the knowledge base.

    Extracts text -> chunks -> embeds -> stores in archival memory.
    """
    from agent.storage.ingestion import ingest_text, ingest_url

    doc_source = state.get("doc_source", "")
    source_type = state.get("doc_source_type", "url")

    if not doc_source:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                content = msg.content
                url_match = re.search(r"https?://\S+", content)
                if url_match:
                    doc_source = url_match.group(0)
                    source_type = "url"
                else:
                    doc_source = content
                    source_type = "text"
                break

    if not doc_source:
        return {"ingest_result": "Error: No document source provided."}

    try:
        if source_type == "url":
            title, chunks = await ingest_url(doc_source)
            if not chunks:
                return {"ingest_result": f"Failed to extract text from URL: {doc_source}"}
            doc_label = title
        else:
            chunks = await ingest_text(
                content=doc_source,
                source_name="manual_text",
                source_type="text",
            )
            if not chunks:
                return {"ingest_result": "No text content to ingest."}
            doc_label = "manual_text"

        entries = [
            {"content": c.content, "metadata": {"source": c.source, "source_type": c.source_type, "chunk_index": c.index, "document": doc_label}}
            for c in chunks
        ]
        entry_ids = await asyncio.to_thread(
            _sync_put_batch_to_archival, entries, "ingested"
        )

        result = (
            f"Ingested '{doc_label}': {len(chunks)} chunks extracted, "
            f"{len(entry_ids)} stored in archival memory. "
            f"Source: {doc_source if source_type == 'url' else 'plain text'}."
        )
        return {
            "ingest_result": result,
            "archival_results": [
                {
                    "id": eid,
                    "content": c.content,
                    "source": c.source,
                    "source_type": c.source_type,
                    "chunk_index": c.index,
                    "document": doc_label,
                }
                for eid, c in zip(entry_ids, chunks)
            ],
        }
    except Exception as e:
        return {"ingest_result": f"Error ingesting document: {e}"}
