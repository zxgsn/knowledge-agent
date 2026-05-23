"""Memory pipeline: extract, dedup, and consolidate conversation facts.

Implements mem0-style pipeline on top of the existing three-layer architecture:
- After each turn: extract facts → dedup against archival → upsert
- Periodically: batch consolidation via embedding clustering
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.db import (
    cleanup_excess,
    cleanup_namespace,
    delete_archival,
    get_all_facts,
    get_existing_memories,
    put_to_archival,
    search_archival_for_dedup,
    update_archival,
)
from agent.prompts import (
    CONSOLIDATION_MERGE_PROMPT,
    MEMORY_JUDGMENT_PROMPT,
    SELECTIVE_EXTRACTION_PROMPT,
)
from agent.state import AgentState
from agent.utils import get_llm, parse_json


# --- Graph nodes ---

async def memory_pipeline(state: AgentState, config: RunnableConfig) -> dict:
    """Selective memory capture: judge → search existing → extract → upsert.

    Two-step LLM pipeline:
    1. MEMORY_JUDGMENT_PROMPT — lightweight check if the turn is worth remembering
    2. SELECTIVE_EXTRACTION_PROMPT — extract ADD/UPDATE/DELETE ops with existing memory context
    """
    turn_count = state.get("turn_count", 0) + 1
    configurable = Configuration.from_runnable_config(config)

    if not configurable.memory_selective_enabled:
        return {"turn_count": turn_count}

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

    llm = get_llm(configurable, temperature=0.0)

    # Step 1: Judge if this turn is worth remembering
    judgment_prompt = MEMORY_JUDGMENT_PROMPT.format(
        user_message=user_msg,
        assistant_message=assistant_msg,
    )
    try:
        judgment_response = await llm.ainvoke(judgment_prompt)
        judgment = parse_json(judgment_response.content)
    except Exception:
        return {"turn_count": turn_count}

    if not judgment.get("memorable", False):
        return {
            "turn_count": turn_count,
            "memory_operations": [{
                "type": "pipeline_judgment",
                "memorable": False,
                "reason": judgment.get("reason", ""),
                "turn_count": turn_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    # Step 2: Search existing memories for context
    search_query = user_msg[:200]
    existing = await asyncio.to_thread(
        get_existing_memories, search_query, "conversation_facts", 10
    )

    existing_text = "None"
    if existing:
        existing_text = "\n".join(
            f"- [id: {m['id']}] {m['content']}" for m in existing
        )

    # Step 3: Extract memory operations
    extraction_prompt = SELECTIVE_EXTRACTION_PROMPT.format(
        existing_memories=existing_text,
        user_message=user_msg,
        assistant_message=assistant_msg,
    )
    try:
        response = await llm.ainvoke(extraction_prompt)
        parsed = parse_json(response.content)
    except Exception:
        return {"turn_count": turn_count}

    operations = parsed.get("memory", [])
    if not operations:
        return {
            "turn_count": turn_count,
            "memory_operations": [{
                "type": "pipeline_extract",
                "facts_count": 0,
                "turn_count": turn_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    # Step 4: Execute operations
    stored_count = 0
    updated_count = 0
    deleted_count = 0
    skipped_count = 0
    fact_log = []

    for op in operations:
        event = op.get("event", "ADD").upper()
        text = op.get("text", "").strip()
        op_id = op.get("id", "")

        if event == "ADD" and text:
            entry_id = await asyncio.to_thread(
                put_to_archival,
                text,
                "conversation_facts",
                {"source": "conversation", "turn": turn_count},
            )
            stored_count += 1
            fact_log.append({"fact": text, "action": "stored", "entry_id": entry_id})

        elif event == "UPDATE" and text and op_id:
            old_memory = op.get("old_memory", "")
            await asyncio.to_thread(
                update_archival,
                op_id,
                text,
                {"source": "memory_pipeline", "old_memory": old_memory[:80]},
            )
            updated_count += 1
            fact_log.append({"fact": text, "action": "updated", "existing_id": op_id})

        elif event == "DELETE" and op_id:
            await asyncio.to_thread(delete_archival, op_id)
            deleted_count += 1
            fact_log.append({"fact": op.get("old_memory", ""), "action": "deleted", "entry_id": op_id})

        else:
            skipped_count += 1
            fact_log.append({"fact": text, "action": "skipped", "reason": f"event:{event}"})

    return {
        "turn_count": turn_count,
        "pipeline_facts": fact_log,
        "memory_operations": [{
            "type": "pipeline_extract",
            "facts_count": len(operations),
            "stored": stored_count,
            "updated": updated_count,
            "deleted": deleted_count,
            "skipped": skipped_count,
            "existing_memories": len(existing),
            "turn_count": turn_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }],
    }


async def _consolidate_namespace(
    namespace: str,
    llm,
    threshold: float = 0.85,
    max_entries: int = 500,
) -> dict:
    """Dedup and merge similar entries in a namespace via embedding clustering."""
    facts = await asyncio.to_thread(get_all_facts, namespace, max_entries)

    if len(facts) < 2:
        return {"namespace": namespace, "action": "skipped", "reason": "too_few", "count": len(facts)}

    import numpy as np

    def parse_vector(text: str) -> list[float]:
        return [float(x) for x in text.strip("[]").split(",")]

    try:
        embeddings = [parse_vector(f["embedding_text"]) for f in facts]
        emb_matrix = np.array(embeddings)
    except Exception:
        return {"namespace": namespace, "action": "skipped", "reason": "embedding_error"}

    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    emb_normed = emb_matrix / norms
    sim_matrix = emb_normed @ emb_normed.T

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

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] >= threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    merged_count = 0
    deleted_count = 0

    for root, members in clusters.items():
        if len(members) < 2:
            continue

        cluster_facts = [facts[m]["content"] for m in members]
        cluster_ids = [facts[m]["id"] for m in members]

        merge_prompt = CONSOLIDATION_MERGE_PROMPT.format(
            entries="\n".join(f"- {f}" for f in cluster_facts)
        )
        try:
            response = await llm.ainvoke(merge_prompt)
            merged_text = response.content.strip()
        except Exception:
            merged_text = max(cluster_facts, key=len)

        for cid in cluster_ids:
            await asyncio.to_thread(delete_archival, cid)
            deleted_count += 1

        await asyncio.to_thread(
            put_to_archival,
            merged_text,
            "conversation_facts",
            {"source": "consolidation", "merged_from": cluster_ids, "original_count": len(members)},
        )
        merged_count += 1

    return {
        "namespace": namespace,
        "total": len(facts),
        "merged": merged_count,
        "deleted": deleted_count,
    }


async def consolidate_memory(state: AgentState, config: RunnableConfig) -> dict:
    """Periodic consolidation: dedup, merge, and cleanup across all managed namespaces."""
    configurable = Configuration.from_runnable_config(config)
    merge_llm = get_llm(configurable, temperature=0.2)

    ops = []
    t0 = datetime.now(timezone.utc)

    # 1. conversation_facts: dedup + merge
    result = await _consolidate_namespace(
        "conversation_facts", merge_llm, threshold=0.85, max_entries=500,
    )
    ops.append({"type": "consolidate", **result, "timestamp": t0.isoformat()})

    # 2. research: dedup + merge
    result = await _consolidate_namespace(
        "research", merge_llm, threshold=0.85, max_entries=200,
    )
    ops.append({"type": "consolidate", **result, "timestamp": t0.isoformat()})

    # 3. ingested: cleanup old + excess (no merge — preserve original chunks)
    cleanup_days = configurable.archival_cleanup_days
    max_entries = configurable.archival_max_entries

    cleaned = await asyncio.to_thread(cleanup_namespace, "ingested", cleanup_days)
    excess = await asyncio.to_thread(cleanup_excess, "ingested", max_entries)
    if cleaned or excess:
        ops.append({
            "type": "cleanup",
            "namespace": "ingested",
            "expired_deleted": cleaned,
            "excess_deleted": excess,
            "timestamp": t0.isoformat(),
        })

    return {"memory_operations": ops}
