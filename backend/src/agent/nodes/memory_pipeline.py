"""Memory pipeline: extract, dedup, and consolidate conversation facts.

Implements mem0-style pipeline on top of the existing three-layer architecture:
- After each turn: extract facts → dedup against archival → upsert
- Periodically: batch consolidation via embedding clustering
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from agent.configuration import Configuration
from agent.prompts import (
    CONSOLIDATION_MERGE_PROMPT,
    FACT_CONFLICT_PROMPT,
    FACT_EXTRACTION_PROMPT,
)
from agent.state import AgentState


# --- Helpers ---

def _get_llm(config: Configuration, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.llm_model,
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        temperature=temperature,
        http_async_client=httpx.AsyncClient(proxy=None),
        extra_body={"thinking": {"type": "enabled"}},
    )


def _parse_json(text: str) -> dict:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# --- Sync DB helpers (run in asyncio.to_thread) ---

def _sync_search_archival_for_dedup(
    query: str, namespace: str = "conversation_facts", limit: int = 3
) -> list[dict]:
    """Search archival for semantically similar facts. Pure cosine similarity."""
    import psycopg
    from pgvector import Vector
    from pgvector.psycopg import register_vector
    from agent.storage import get_db_url, get_embeddings

    try:
        embeddings = get_embeddings()
        query_embedding = Vector(embeddings.embed_query(query))
    except Exception:
        return []

    try:
        conn = psycopg.connect(get_db_url(), connect_timeout=5)
    except Exception:
        return []

    register_vector(conn)
    rows = conn.execute(
        """
        SELECT id, content, metadata,
               1 - (embedding <=> %s::vector) AS score
        FROM archival_memory
        WHERE namespace = %s
          AND 1 - (embedding <=> %s::vector) > 0.5
        ORDER BY score DESC
        LIMIT %s
        """,
        (query_embedding, namespace, query_embedding, int(limit)),
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        score = float(row[3])
        meta = row[2] if isinstance(row[2], dict) else json.loads(row[2])
        results.append({"id": row[0], "content": row[1], "metadata": meta, "score": score})
    return results


def _sync_put_fact_to_archival(content: str, metadata: dict) -> str:
    """Store a new fact in archival memory under conversation_facts namespace."""
    import psycopg
    from pgvector import Vector
    from pgvector.psycopg import register_vector
    from agent.storage import get_db_url, get_embeddings

    embeddings = get_embeddings()
    entry_id = str(uuid.uuid4())
    metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
    metadata["type"] = "extracted_fact"
    embedding = Vector(embeddings.embed_query(content))

    conn = psycopg.connect(get_db_url(), connect_timeout=5)
    register_vector(conn)
    conn.execute(
        "INSERT INTO archival_memory (id, namespace, content, metadata, embedding) "
        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
        (entry_id, "conversation_facts", content, json.dumps(metadata), embedding),
    )
    conn.commit()
    conn.close()
    return entry_id


def _sync_update_archival(entry_id: str, content: str, metadata: dict) -> None:
    """Update an existing archival entry's content and embedding."""
    import psycopg
    from pgvector import Vector
    from pgvector.psycopg import register_vector
    from agent.storage import get_db_url, get_embeddings

    embeddings = get_embeddings()
    embedding = Vector(embeddings.embed_query(content))
    metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
    metadata["updated_by"] = "memory_pipeline"

    conn = psycopg.connect(get_db_url(), connect_timeout=5)
    register_vector(conn)
    conn.execute(
        "UPDATE archival_memory SET content = %s, metadata = %s, embedding = %s WHERE id = %s",
        (content, json.dumps(metadata), embedding, entry_id),
    )
    conn.commit()
    conn.close()


def _sync_delete_archival(entry_id: str) -> bool:
    """Delete an archival entry by ID."""
    import psycopg
    from agent.storage import get_db_url

    conn = psycopg.connect(get_db_url(), connect_timeout=5)
    result = conn.execute("DELETE FROM archival_memory WHERE id = %s", (entry_id,))
    conn.commit()
    conn.close()
    return result.rowcount > 0


def _sync_get_all_facts(namespace: str = "conversation_facts", limit: int = 500) -> list[dict]:
    """Fetch all facts in a namespace for consolidation."""
    import psycopg
    from pgvector.psycopg import register_vector
    from agent.storage import get_db_url

    try:
        conn = psycopg.connect(get_db_url(), connect_timeout=5)
    except Exception:
        return []

    register_vector(conn)
    rows = conn.execute(
        "SELECT id, content, metadata, embedding::text FROM archival_memory "
        "WHERE namespace = %s ORDER BY created_at DESC LIMIT %s",
        (namespace, int(limit)),
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        meta = row[2] if isinstance(row[2], dict) else json.loads(row[2])
        results.append({"id": row[0], "content": row[1], "metadata": meta, "embedding_text": row[3]})
    return results


# --- Graph nodes ---

async def memory_pipeline(state: AgentState, config: RunnableConfig) -> dict:
    """Extract facts from the latest exchange and store in archival memory."""
    configurable = Configuration.from_runnable_config(config)
    turn_count = state.get("turn_count", 0) + 1

    # Extract last user + assistant messages
    messages = state["messages"]
    user_msg = ""
    assistant_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not assistant_msg:
            assistant_msg = msg.content
        elif isinstance(msg, HumanMessage) and msg.content and not user_msg:
            user_msg = msg.content
        if user_msg and assistant_msg:
            break

    if not user_msg or not assistant_msg:
        return {"turn_count": turn_count}

    # Extract facts via LLM
    llm = _get_llm(configurable, temperature=0.0)
    extraction_prompt = FACT_EXTRACTION_PROMPT.format(
        user_message=user_msg, assistant_message=assistant_msg
    )
    response = await llm.ainvoke(extraction_prompt)
    parsed = _parse_json(response.content)
    facts = parsed.get("facts", [])

    if not facts:
        return {
            "turn_count": turn_count,
            "memory_operations": [{
                "type": "pipeline_extract",
                "facts_count": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    # Dedup + upsert each fact
    conflict_llm = _get_llm(configurable, temperature=0.0)
    dedup_threshold = configurable.memory_dedup_threshold
    stored_count = 0
    updated_count = 0
    skipped_count = 0
    fact_log = []

    for fact in facts:
        similar = await asyncio.to_thread(
            _sync_search_archival_for_dedup, fact, "conversation_facts", 3
        )

        conflict = None
        for s in similar:
            if s["score"] >= dedup_threshold:
                conflict = s
                break

        if conflict:
            conflict_prompt = FACT_CONFLICT_PROMPT.format(
                new_fact=fact,
                existing_fact=conflict["content"],
                score=f"{conflict['score']:.2f}",
            )
            conflict_response = await conflict_llm.ainvoke(conflict_prompt)
            decision = _parse_json(conflict_response.content)

            if decision.get("decision") == "update" and decision.get("merged"):
                await asyncio.to_thread(
                    _sync_update_archival,
                    conflict["id"],
                    decision["merged"],
                    {"source": "memory_pipeline", "merged_from": fact[:50]},
                )
                updated_count += 1
                fact_log.append({"fact": fact, "action": "updated", "existing_id": conflict["id"]})
            else:
                skipped_count += 1
                fact_log.append({"fact": fact, "action": "skipped", "reason": decision.get("reason", "duplicate")})
        else:
            entry_id = await asyncio.to_thread(
                _sync_put_fact_to_archival,
                fact,
                {"source": "conversation", "turn": turn_count},
            )
            stored_count += 1
            fact_log.append({"fact": fact, "action": "stored", "entry_id": entry_id})

    memory_ops = [{
        "type": "pipeline_extract",
        "facts_extracted": len(facts),
        "stored": stored_count,
        "updated": updated_count,
        "skipped": skipped_count,
        "turn_count": turn_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }]

    return {
        "turn_count": turn_count,
        "pipeline_facts": fact_log,
        "memory_operations": memory_ops,
    }


async def consolidate_memory(state: AgentState, config: RunnableConfig) -> dict:
    """Periodic consolidation: dedup and merge similar facts via embedding clustering."""
    configurable = Configuration.from_runnable_config(config)

    facts = await asyncio.to_thread(_sync_get_all_facts, "conversation_facts", 500)

    if len(facts) < 2:
        return {
            "memory_operations": [{
                "type": "consolidate",
                "action": "skipped",
                "reason": "too_few_facts",
                "count": len(facts),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    # Parse stored embeddings (avoid re-embedding)
    import numpy as np

    def parse_vector(text: str) -> list[float]:
        return [float(x) for x in text.strip("[]").split(",")]

    try:
        embeddings = [parse_vector(f["embedding_text"]) for f in facts]
        emb_matrix = np.array(embeddings)
    except Exception:
        return {
            "memory_operations": [{
                "type": "consolidate",
                "action": "skipped",
                "reason": "embedding_parse_error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    # Normalize and compute pairwise cosine similarity
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    emb_normed = emb_matrix / norms
    sim_matrix = emb_normed @ emb_normed.T

    # Union-Find clustering
    n = len(facts)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    THRESHOLD = 0.85
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] >= THRESHOLD:
                union(i, j)

    # Group by cluster
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    # Merge clusters with 2+ members
    merge_llm = _get_llm(configurable, temperature=0.2)
    merged_count = 0
    deleted_ids = []

    for root, members in clusters.items():
        if len(members) < 2:
            continue

        cluster_facts = [facts[m]["content"] for m in members]
        cluster_ids = [facts[m]["id"] for m in members]

        merge_prompt = CONSOLIDATION_MERGE_PROMPT.format(
            entries="\n".join(f"- {f}" for f in cluster_facts)
        )
        response = await merge_llm.ainvoke(merge_prompt)
        merged_text = response.content.strip()

        for cid in cluster_ids:
            await asyncio.to_thread(_sync_delete_archival, cid)
            deleted_ids.append(cid)

        await asyncio.to_thread(
            _sync_put_fact_to_archival,
            merged_text,
            {"source": "consolidation", "merged_from": cluster_ids, "original_count": len(members)},
        )
        merged_count += 1

    return {
        "memory_operations": [{
            "type": "consolidate",
            "total_facts": len(facts),
            "clusters_merged": merged_count,
            "entries_deleted": len(deleted_ids),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }],
    }
